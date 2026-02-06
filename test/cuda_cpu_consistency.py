#!/usr/bin/env python3

import argparse
import os
import pathlib
import subprocess
import tempfile


def write_tiny_edgelist(path: pathlib.Path, n_vertices: int = 48) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Synthetic connected graph for CPU/GPU consistency checks\n")
        for i in range(n_vertices):
            a = i + 1
            b = ((i + 1) % n_vertices) + 1
            c = ((i + 5) % n_vertices) + 1
            d = ((i + 11) % n_vertices) + 1
            handle.write(f"V{a} V{b}\n")
            handle.write(f"V{a} V{c}\n")
            handle.write(f"V{a} V{d}\n")


def run_mercator(binary: pathlib.Path, edge_path: pathlib.Path, out_root: pathlib.Path, seed: int, disable_cuda: bool) -> None:
    env = os.environ.copy()
    env["DMERCATOR_TRACE_LOGLIKELIHOOD"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    if disable_cuda:
        env["DMERCATOR_DISABLE_CUDA"] = "1"
    cmd = [
        str(binary),
        "-q",
        "-s",
        str(seed),
        "-d",
        "2",
        "-o",
        str(out_root),
        str(edge_path),
    ]
    subprocess.run(cmd, check=True, env=env)


def parse_inf_coord(path: pathlib.Path):
    beta = None
    mu = None
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                if stripped.startswith("#   - beta:"):
                    beta = float(stripped.split(":")[-1].strip())
                elif stripped.startswith("#   - mu:"):
                    mu = float(stripped.split(":")[-1].strip())
                continue
            parts = stripped.split()
            rows[parts[0]] = [float(value) for value in parts[1:]]
    if beta is None or mu is None:
        raise RuntimeError(f"Could not parse beta/mu from {path}")
    return beta, mu, rows


def parse_ll_trace(path: pathlib.Path):
    if not path.exists():
        raise RuntimeError(f"Missing likelihood trace: {path}")
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            _, ll = stripped.split()
            values.append(float(ll))
    if not values:
        raise RuntimeError(f"Likelihood trace is empty: {path}")
    return values


def compare_cpu_gpu(cpu_root: pathlib.Path,
                    gpu_root: pathlib.Path,
                    param_tol: float,
                    node_tol: float,
                    coord_tol: float,
                    trace_rel_tol: float) -> None:
    cpu_beta, cpu_mu, cpu_rows = parse_inf_coord(cpu_root.with_suffix(".inf_coord"))
    gpu_beta, gpu_mu, gpu_rows = parse_inf_coord(gpu_root.with_suffix(".inf_coord"))

    if abs(cpu_beta - gpu_beta) > param_tol:
        raise AssertionError(f"beta mismatch: cpu={cpu_beta}, gpu={gpu_beta}, tol={param_tol}")
    if abs(cpu_mu - gpu_mu) > param_tol:
        raise AssertionError(f"mu mismatch: cpu={cpu_mu}, gpu={gpu_mu}, tol={param_tol}")

    cpu_nodes = sorted(cpu_rows.keys())
    gpu_nodes = sorted(gpu_rows.keys())
    if cpu_nodes != gpu_nodes:
        raise AssertionError("Node sets differ between CPU and GPU outputs")

    max_node_scalar_diff = 0.0
    for node in cpu_nodes:
        cpu_values = cpu_rows[node]
        gpu_values = gpu_rows[node]
        if len(cpu_values) != len(gpu_values):
            raise AssertionError(f"Coordinate width mismatch for node {node}")
        for idx in range(min(2, len(cpu_values))):
            diff = abs(cpu_values[idx] - gpu_values[idx])
            max_node_scalar_diff = max(max_node_scalar_diff, diff)
            if diff > node_tol:
                raise AssertionError(
                    f"Node scalar mismatch at {node}[{idx}]: cpu={cpu_values[idx]}, gpu={gpu_values[idx]}, tol={node_tol}"
                )

    # Compare pairwise dot products of positional coordinates (rotation-invariant).
    max_pairwise_diff = 0.0
    sample_nodes = cpu_nodes[: min(24, len(cpu_nodes))]
    for i, ni in enumerate(sample_nodes):
        cvi = cpu_rows[ni][2:]
        gvi = gpu_rows[ni][2:]
        for nj in sample_nodes[i + 1:]:
            cvj = cpu_rows[nj][2:]
            gvj = gpu_rows[nj][2:]
            cpu_dot = sum(a * b for a, b in zip(cvi, cvj))
            gpu_dot = sum(a * b for a, b in zip(gvi, gvj))
            diff = abs(cpu_dot - gpu_dot)
            max_pairwise_diff = max(max_pairwise_diff, diff)
            if diff > coord_tol:
                raise AssertionError(
                    f"Pairwise coordinate mismatch ({ni},{nj}): cpu_dot={cpu_dot}, gpu_dot={gpu_dot}, tol={coord_tol}"
                )

    cpu_trace = parse_ll_trace(cpu_root.with_suffix(".inf_ll_trace"))
    gpu_trace = parse_ll_trace(gpu_root.with_suffix(".inf_ll_trace"))
    if len(cpu_trace) != len(gpu_trace):
        raise AssertionError(
            f"Likelihood trace length mismatch: cpu={len(cpu_trace)}, gpu={len(gpu_trace)}"
        )

    max_trace_rel_err = 0.0
    for idx, (cpu_ll, gpu_ll) in enumerate(zip(cpu_trace, gpu_trace)):
        denom = max(1.0, abs(cpu_ll), abs(gpu_ll))
        rel_err = abs(cpu_ll - gpu_ll) / denom
        max_trace_rel_err = max(max_trace_rel_err, rel_err)
        if rel_err > trace_rel_tol:
            raise AssertionError(
                f"Likelihood trace mismatch at step {idx}: cpu={cpu_ll}, gpu={gpu_ll}, rel_err={rel_err}, tol={trace_rel_tol}"
            )

    print("CPU/GPU consistency check passed")
    print(f"  max |beta,mu| diff: {max(abs(cpu_beta - gpu_beta), abs(cpu_mu - gpu_mu)):.6g}")
    print(f"  max node scalar diff (kappa/radial): {max_node_scalar_diff:.6g}")
    print(f"  max pairwise dot-product diff: {max_pairwise_diff:.6g}")
    print(f"  max relative ll-trace diff: {max_trace_rel_err:.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare D-Mercator CPU and CUDA runs on a tiny graph.")
    parser.add_argument("--cpu-bin", required=True, help="Path to CPU-only mercator binary")
    parser.add_argument("--gpu-bin", required=True, help="Path to CUDA-enabled mercator binary")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--param-tol", type=float, default=5e-2, help="Absolute tolerance for beta and mu")
    parser.add_argument("--node-tol", type=float, default=2e-1, help="Absolute tolerance for kappa/radial values")
    parser.add_argument("--coord-tol", type=float, default=2e-1, help="Absolute tolerance for pairwise dot products")
    parser.add_argument("--trace-rel-tol", type=float, default=5e-3, help="Relative tolerance for log-likelihood trace")
    args = parser.parse_args()

    cpu_bin = pathlib.Path(args.cpu_bin).resolve()
    gpu_bin = pathlib.Path(args.gpu_bin).resolve()
    if not cpu_bin.exists():
        raise FileNotFoundError(f"CPU binary does not exist: {cpu_bin}")
    if not gpu_bin.exists():
        raise FileNotFoundError(f"GPU binary does not exist: {gpu_bin}")

    with tempfile.TemporaryDirectory(prefix="dmercator_cuda_test_") as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        edge_path = tmp_path / "tiny.edge"
        write_tiny_edgelist(edge_path)

        cpu_root = tmp_path / "cpu_run"
        gpu_root = tmp_path / "gpu_run"

        same_binary = cpu_bin == gpu_bin
        run_mercator(cpu_bin, edge_path, cpu_root, args.seed, disable_cuda=same_binary)
        run_mercator(gpu_bin, edge_path, gpu_root, args.seed, disable_cuda=False)

        compare_cpu_gpu(
            cpu_root=cpu_root,
            gpu_root=gpu_root,
            param_tol=args.param_tol,
            node_tol=args.node_tol,
            coord_tol=args.coord_tol,
            trace_rel_tol=args.trace_rel_tol,
        )


if __name__ == "__main__":
    main()
