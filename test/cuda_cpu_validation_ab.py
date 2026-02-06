#!/usr/bin/env python3

import argparse
import os
import pathlib
import subprocess
import tempfile


REQUIRED_VALIDATION_SUFFIXES = [
    ".inf_pconn",
    ".inf_theta_density",
    ".inf_vprop",
    ".inf_vstat",
    ".obs_vstat",
]


def write_tiny_edgelist(path: pathlib.Path, n_vertices: int = 56) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Synthetic connected graph for validation CPU-GPU A/B checks\n")
        for i in range(n_vertices):
            a = i + 1
            b = ((i + 1) % n_vertices) + 1
            c = ((i + 7) % n_vertices) + 1
            d = ((i + 13) % n_vertices) + 1
            handle.write(f"V{a} V{b}\n")
            handle.write(f"V{a} V{c}\n")
            handle.write(f"V{a} V{d}\n")


def run_mercator(
    binary: pathlib.Path,
    edge_path: pathlib.Path,
    out_root: pathlib.Path,
    seed: int,
    disable_cuda: bool,
    extra_args: list[str] | None = None,
) -> None:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    if disable_cuda:
        env["DMERCATOR_DISABLE_CUDA"] = "1"
    else:
        env.pop("DMERCATOR_DISABLE_CUDA", None)

    cmd = [
        str(binary),
        "-q",
        "-s",
        str(seed),
        "-d",
        "1",
        "-o",
        str(out_root),
    ]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(str(edge_path))
    subprocess.run(cmd, check=True, env=env)


def read_lines(path: pathlib.Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing output file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def split_header_and_data(lines: list[str]) -> tuple[list[str], list[str]]:
    header: list[str] = []
    data: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            header.append(line)
        else:
            data.append(line)
    return header, data


def parse_dense_numeric_rows(path: pathlib.Path) -> tuple[list[str], list[list[float]]]:
    lines = read_lines(path)
    header, data = split_header_and_data(lines)
    rows: list[list[float]] = []
    for line in data:
        parts = line.split()
        rows.append([float(value) for value in parts])
    return header, rows


def parse_keyed_numeric_rows(path: pathlib.Path, key_type) -> tuple[list[str], dict]:
    lines = read_lines(path)
    header, data = split_header_and_data(lines)
    rows = {}
    for line in data:
        parts = line.split()
        key = key_type(parts[0])
        rows[key] = [float(value) for value in parts[1:]]
    return header, rows


def diff_ok(lhs: float, rhs: float, abs_tol: float, rel_tol: float) -> bool:
    abs_diff = abs(lhs - rhs)
    if abs_diff <= abs_tol:
        return True
    denom = max(1.0, abs(lhs), abs(rhs))
    rel_diff = abs_diff / denom
    return rel_diff <= rel_tol


def compare_headers_and_shape(cpu_path: pathlib.Path, gpu_path: pathlib.Path) -> None:
    cpu_lines = read_lines(cpu_path)
    gpu_lines = read_lines(gpu_path)
    cpu_header, cpu_data = split_header_and_data(cpu_lines)
    gpu_header, gpu_data = split_header_and_data(gpu_lines)

    if cpu_header != gpu_header:
        raise AssertionError(f"Header mismatch for {cpu_path.name}")
    if len(cpu_data) != len(gpu_data):
        raise AssertionError(
            f"Data row count mismatch for {cpu_path.name}: cpu={len(cpu_data)}, gpu={len(gpu_data)}"
        )

    for idx, (cpu_row, gpu_row) in enumerate(zip(cpu_data, gpu_data)):
        cpu_cols = len(cpu_row.split())
        gpu_cols = len(gpu_row.split())
        if cpu_cols != gpu_cols:
            raise AssertionError(
                f"Column count mismatch for {cpu_path.name} row {idx}: cpu={cpu_cols}, gpu={gpu_cols}"
            )


def compare_dense_table(
    cpu_path: pathlib.Path,
    gpu_path: pathlib.Path,
    abs_tol: float,
    rel_tol: float,
) -> float:
    cpu_header, cpu_rows = parse_dense_numeric_rows(cpu_path)
    gpu_header, gpu_rows = parse_dense_numeric_rows(gpu_path)
    if cpu_header != gpu_header:
        raise AssertionError(f"Header mismatch for {cpu_path.name}")
    if len(cpu_rows) != len(gpu_rows):
        raise AssertionError(
            f"Data row count mismatch for {cpu_path.name}: cpu={len(cpu_rows)}, gpu={len(gpu_rows)}"
        )

    max_abs_diff = 0.0
    for ridx, (cpu_row, gpu_row) in enumerate(zip(cpu_rows, gpu_rows)):
        if len(cpu_row) != len(gpu_row):
            raise AssertionError(
                f"Column mismatch for {cpu_path.name} row {ridx}: cpu={len(cpu_row)}, gpu={len(gpu_row)}"
            )
        for cidx, (cpu_value, gpu_value) in enumerate(zip(cpu_row, gpu_row)):
            abs_diff = abs(cpu_value - gpu_value)
            max_abs_diff = max(max_abs_diff, abs_diff)
            if not diff_ok(cpu_value, gpu_value, abs_tol, rel_tol):
                raise AssertionError(
                    f"Mismatch in {cpu_path.name} at row={ridx} col={cidx}: "
                    f"cpu={cpu_value}, gpu={gpu_value}, abs_tol={abs_tol}, rel_tol={rel_tol}"
                )
    return max_abs_diff


def compare_vprop(
    cpu_path: pathlib.Path,
    gpu_path: pathlib.Path,
    abs_tol: float,
    rel_tol: float,
) -> float:
    cpu_header, cpu_rows = parse_keyed_numeric_rows(cpu_path, str)
    gpu_header, gpu_rows = parse_keyed_numeric_rows(gpu_path, str)
    if cpu_header != gpu_header:
        raise AssertionError("Header mismatch for .inf_vprop")
    if set(cpu_rows.keys()) != set(gpu_rows.keys()):
        raise AssertionError("Vertex keys differ for .inf_vprop")

    max_abs_diff = 0.0
    for node in sorted(cpu_rows.keys()):
        cpu_values = cpu_rows[node]
        gpu_values = gpu_rows[node]
        if len(cpu_values) != len(gpu_values):
            raise AssertionError(f"Column mismatch for .inf_vprop node {node}")
        for cidx, (cpu_value, gpu_value) in enumerate(zip(cpu_values, gpu_values)):
            abs_diff = abs(cpu_value - gpu_value)
            max_abs_diff = max(max_abs_diff, abs_diff)
            if not diff_ok(cpu_value, gpu_value, abs_tol, rel_tol):
                raise AssertionError(
                    f".inf_vprop mismatch at node={node} col={cidx}: "
                    f"cpu={cpu_value}, gpu={gpu_value}, abs_tol={abs_tol}, rel_tol={rel_tol}"
                )
    return max_abs_diff


def summarize_inf_vstat(rows: dict[int, list[float]]) -> dict[str, float]:
    degree_mass = 0.0
    mean_degree = 0.0
    mean_neighbor_degree = 0.0
    mean_clustering = 0.0
    for degree, values in rows.items():
        deg_dist = values[0]
        avg_deg_n = values[6]
        clust = values[10]
        degree_mass += deg_dist
        mean_degree += degree * deg_dist
        mean_neighbor_degree += deg_dist * avg_deg_n
        mean_clustering += deg_dist * clust
    return {
        "degree_mass": degree_mass,
        "mean_degree": mean_degree,
        "mean_neighbor_degree": mean_neighbor_degree,
        "mean_clustering": mean_clustering,
    }


def compare_vstat(
    cpu_path: pathlib.Path,
    gpu_path: pathlib.Path,
    per_entry_abs_tol: float,
    per_entry_rel_tol: float,
    summary_abs_tol: float,
) -> tuple[float, dict[str, float]]:
    cpu_header, cpu_rows = parse_keyed_numeric_rows(cpu_path, int)
    gpu_header, gpu_rows = parse_keyed_numeric_rows(gpu_path, int)
    if cpu_header != gpu_header:
        raise AssertionError("Header mismatch for .inf_vstat")

    common_keys = set(cpu_rows.keys()) & set(gpu_rows.keys())
    if not common_keys:
        raise AssertionError("No overlapping degree classes in .inf_vstat")

    max_abs_diff = 0.0
    for degree in sorted(common_keys):
        cpu_values = cpu_rows[degree]
        gpu_values = gpu_rows[degree]
        if len(cpu_values) != len(gpu_values):
            raise AssertionError(f"Column mismatch for .inf_vstat degree {degree}")
        for cidx, (cpu_value, gpu_value) in enumerate(zip(cpu_values, gpu_values)):
            abs_diff = abs(cpu_value - gpu_value)
            max_abs_diff = max(max_abs_diff, abs_diff)
            if not diff_ok(cpu_value, gpu_value, per_entry_abs_tol, per_entry_rel_tol):
                raise AssertionError(
                    f".inf_vstat mismatch at degree={degree} col={cidx}: "
                    f"cpu={cpu_value}, gpu={gpu_value}, "
                    f"abs_tol={per_entry_abs_tol}, rel_tol={per_entry_rel_tol}"
                )

    cpu_summary = summarize_inf_vstat(cpu_rows)
    gpu_summary = summarize_inf_vstat(gpu_rows)
    summary_diffs = {}
    for key in cpu_summary.keys():
        diff = abs(cpu_summary[key] - gpu_summary[key])
        summary_diffs[key] = diff
        if diff > summary_abs_tol:
            raise AssertionError(
                f".inf_vstat summary mismatch for {key}: "
                f"cpu={cpu_summary[key]}, gpu={gpu_summary[key]}, abs_tol={summary_abs_tol}"
            )
    return max_abs_diff, summary_diffs


def compare_obs_vstat_exact(cpu_path: pathlib.Path, gpu_path: pathlib.Path) -> None:
    cpu_lines = read_lines(cpu_path)
    gpu_lines = read_lines(gpu_path)
    if cpu_lines != gpu_lines:
        raise AssertionError(".obs_vstat differs between CPU and GPU runs")


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-vs-GPU A/B checks for validation mode (-v).")
    parser.add_argument("--cpu-bin", required=True, help="Path to CPU-only mercator binary")
    parser.add_argument("--gpu-bin", required=True, help="Path to CUDA-enabled mercator binary")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--pconn-abs-tol", type=float, default=5e-8)
    parser.add_argument("--pconn-rel-tol", type=float, default=5e-6)
    parser.add_argument("--theta-abs-tol", type=float, default=5e-8)
    parser.add_argument("--theta-rel-tol", type=float, default=5e-6)
    parser.add_argument("--vprop-abs-tol", type=float, default=0.35)
    parser.add_argument("--vprop-rel-tol", type=float, default=0.15)
    parser.add_argument("--vstat-entry-abs-tol", type=float, default=0.2)
    parser.add_argument("--vstat-entry-rel-tol", type=float, default=0.2)
    parser.add_argument("--vstat-summary-abs-tol", type=float, default=0.12)
    args = parser.parse_args()

    cpu_bin = pathlib.Path(args.cpu_bin).resolve()
    gpu_bin = pathlib.Path(args.gpu_bin).resolve()
    if not cpu_bin.exists():
        raise FileNotFoundError(f"CPU binary does not exist: {cpu_bin}")
    if not gpu_bin.exists():
        raise FileNotFoundError(f"GPU binary does not exist: {gpu_bin}")

    with tempfile.TemporaryDirectory(prefix="dmercator_validation_ab_") as tmp_dir:
        tmp = pathlib.Path(tmp_dir)
        edge_path = tmp / "tiny.edge"
        write_tiny_edgelist(edge_path)

        # Build a common reference coordinate file, then run validation-only mode from it.
        ref_root = tmp / "ref"
        run_mercator(
            binary=cpu_bin,
            edge_path=edge_path,
            out_root=ref_root,
            seed=args.seed,
            disable_cuda=True,
            extra_args=[],
        )
        ref_coord = ref_root.with_suffix(".inf_coord")
        if not ref_coord.exists():
            raise FileNotFoundError(f"Reference coordinates were not generated: {ref_coord}")

        cpu_root = tmp / "cpu_validation"
        gpu_root = tmp / "gpu_validation"
        same_binary = cpu_bin == gpu_bin

        validation_args = ["-f", "-k", "-r", str(ref_coord), "-v"]
        run_mercator(
            binary=cpu_bin,
            edge_path=edge_path,
            out_root=cpu_root,
            seed=args.seed,
            disable_cuda=same_binary,
            extra_args=validation_args,
        )
        run_mercator(
            binary=gpu_bin,
            edge_path=edge_path,
            out_root=gpu_root,
            seed=args.seed,
            disable_cuda=False,
            extra_args=validation_args,
        )

        for suffix in REQUIRED_VALIDATION_SUFFIXES:
            compare_headers_and_shape(cpu_root.with_suffix(suffix), gpu_root.with_suffix(suffix))

        pconn_max = compare_dense_table(
            cpu_root.with_suffix(".inf_pconn"),
            gpu_root.with_suffix(".inf_pconn"),
            abs_tol=args.pconn_abs_tol,
            rel_tol=args.pconn_rel_tol,
        )
        theta_max = compare_dense_table(
            cpu_root.with_suffix(".inf_theta_density"),
            gpu_root.with_suffix(".inf_theta_density"),
            abs_tol=args.theta_abs_tol,
            rel_tol=args.theta_rel_tol,
        )
        vprop_max = compare_vprop(
            cpu_root.with_suffix(".inf_vprop"),
            gpu_root.with_suffix(".inf_vprop"),
            abs_tol=args.vprop_abs_tol,
            rel_tol=args.vprop_rel_tol,
        )
        vstat_max, vstat_summary_diffs = compare_vstat(
            cpu_root.with_suffix(".inf_vstat"),
            gpu_root.with_suffix(".inf_vstat"),
            per_entry_abs_tol=args.vstat_entry_abs_tol,
            per_entry_rel_tol=args.vstat_entry_rel_tol,
            summary_abs_tol=args.vstat_summary_abs_tol,
        )
        compare_obs_vstat_exact(
            cpu_root.with_suffix(".obs_vstat"),
            gpu_root.with_suffix(".obs_vstat"),
        )

        print("CPU/GPU validation A/B checks passed")
        print(f"  max .inf_pconn abs diff: {pconn_max:.6g}")
        print(f"  max .inf_theta_density abs diff: {theta_max:.6g}")
        print(f"  max .inf_vprop abs diff: {vprop_max:.6g}")
        print(f"  max .inf_vstat per-entry abs diff: {vstat_max:.6g}")
        print(
            "  .inf_vstat summary diffs: "
            f"degree_mass={vstat_summary_diffs['degree_mass']:.6g}, "
            f"mean_degree={vstat_summary_diffs['mean_degree']:.6g}, "
            f"mean_neighbor_degree={vstat_summary_diffs['mean_neighbor_degree']:.6g}, "
            f"mean_clustering={vstat_summary_diffs['mean_clustering']:.6g}"
        )


if __name__ == "__main__":
    main()
