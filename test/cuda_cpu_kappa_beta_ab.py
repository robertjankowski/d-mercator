#!/usr/bin/env python3

import argparse
import os
import pathlib
import shutil
import subprocess
import tempfile


def write_tiny_edgelist(path: pathlib.Path, n_vertices: int = 56) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Synthetic connected graph for kappa/beta CPU-GPU A/B checks\n")
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
    dim: int,
    disable_cuda: bool,
    extra_args: list[str] | None = None,
    expected_return_codes: tuple[int, ...] = (0,),
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
        str(dim),
        "-o",
        str(out_root),
    ]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(str(edge_path))

    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode not in expected_return_codes:
        raise RuntimeError(
            f"Command failed (rc={proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def parse_inf_coord(path: pathlib.Path) -> tuple[float, dict[str, float]]:
    beta = None
    kappas: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                if stripped.startswith("#   - beta:"):
                    beta = float(stripped.split(":")[-1].strip())
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            kappas[parts[0]] = float(parts[1])
    if beta is None:
        raise RuntimeError(f"Could not parse beta from {path}")
    return beta, kappas


def parse_kappas(path: pathlib.Path) -> list[float]:
    if not path.exists():
        raise RuntimeError(f"Missing kappas file: {path}")
    values: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                values.append(float(stripped))
    if not values:
        raise RuntimeError(f"Kappas file is empty: {path}")
    return values


def compare_dict_kappas(cpu_kappa: dict[str, float], gpu_kappa: dict[str, float], tol: float) -> float:
    if set(cpu_kappa.keys()) != set(gpu_kappa.keys()):
        raise AssertionError("Node sets differ between CPU and GPU inferred coordinates")
    max_diff = 0.0
    for node in sorted(cpu_kappa.keys()):
        diff = abs(cpu_kappa[node] - gpu_kappa[node])
        max_diff = max(max_diff, diff)
        if diff > tol:
            raise AssertionError(
                f"kappa mismatch for {node}: cpu={cpu_kappa[node]}, gpu={gpu_kappa[node]}, tol={tol}"
            )
    return max_diff


def compare_list_kappas(cpu_kappa: list[float], gpu_kappa: list[float], tol: float) -> float:
    if len(cpu_kappa) != len(gpu_kappa):
        raise AssertionError(f"Kappa vector length mismatch: cpu={len(cpu_kappa)}, gpu={len(gpu_kappa)}")
    max_diff = 0.0
    for idx, (cpu_value, gpu_value) in enumerate(zip(cpu_kappa, gpu_kappa)):
        diff = abs(cpu_value - gpu_value)
        max_diff = max(max_diff, diff)
        if diff > tol:
            raise AssertionError(
                f"kappa-only mismatch at index {idx}: cpu={cpu_value}, gpu={gpu_value}, tol={tol}"
            )
    return max_diff


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-vs-GPU A/B checks for beta and kappa inference paths.")
    parser.add_argument("--cpu-bin", required=True, help="Path to CPU-only mercator binary")
    parser.add_argument("--gpu-bin", required=True, help="Path to CUDA-enabled mercator binary")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--dimension", type=int, default=1)
    parser.add_argument("--beta-tol", type=float, default=6e-2)
    parser.add_argument("--kappa-tol", type=float, default=2e-1)
    args = parser.parse_args()

    cpu_bin = pathlib.Path(args.cpu_bin).resolve()
    gpu_bin = pathlib.Path(args.gpu_bin).resolve()
    if not cpu_bin.exists():
        raise FileNotFoundError(f"CPU binary does not exist: {cpu_bin}")
    if not gpu_bin.exists():
        raise FileNotFoundError(f"GPU binary does not exist: {gpu_bin}")

    with tempfile.TemporaryDirectory(prefix="dmercator_kappa_beta_ab_") as tmp_dir:
        tmp = pathlib.Path(tmp_dir)

        # Stage 1: beta + kappa post-processing A/B test (-p).
        edge_path = tmp / "tiny.edge"
        write_tiny_edgelist(edge_path)
        cpu_root = tmp / "cpu_beta_kappa"
        gpu_root = tmp / "gpu_beta_kappa"
        same_binary = cpu_bin == gpu_bin

        run_mercator(
            binary=cpu_bin,
            edge_path=edge_path,
            out_root=cpu_root,
            seed=args.seed,
            dim=args.dimension,
            disable_cuda=same_binary,
            extra_args=["-p"],
            expected_return_codes=(0,),
        )
        run_mercator(
            binary=gpu_bin,
            edge_path=edge_path,
            out_root=gpu_root,
            seed=args.seed,
            dim=args.dimension,
            disable_cuda=False,
            extra_args=["-p"],
            expected_return_codes=(0,),
        )

        cpu_beta, cpu_kappa = parse_inf_coord(cpu_root.with_suffix(".inf_coord"))
        gpu_beta, gpu_kappa = parse_inf_coord(gpu_root.with_suffix(".inf_coord"))

        beta_diff = abs(cpu_beta - gpu_beta)
        if beta_diff > args.beta_tol:
            raise AssertionError(f"beta mismatch: cpu={cpu_beta}, gpu={gpu_beta}, tol={args.beta_tol}")
        max_kappa_diff_post = compare_dict_kappas(cpu_kappa, gpu_kappa, args.kappa_tol)

        # Stage 2: kappa-only inference A/B test (-e with fixed beta and fixed positions).
        ref_inf_coord = cpu_root.with_suffix(".inf_coord")
        beta_for_kappa_only = cpu_beta

        edge_cpu = tmp / "tiny_cpu.edge"
        edge_gpu = tmp / "tiny_gpu.edge"
        shutil.copyfile(edge_path, edge_cpu)
        shutil.copyfile(edge_path, edge_gpu)

        run_mercator(
            binary=cpu_bin,
            edge_path=edge_cpu,
            out_root=tmp / "cpu_kappa_only",
            seed=args.seed,
            dim=args.dimension,
            disable_cuda=same_binary,
            extra_args=["-e", "-r", str(ref_inf_coord), "-b", str(beta_for_kappa_only)],
            expected_return_codes=(13,),
        )
        run_mercator(
            binary=gpu_bin,
            edge_path=edge_gpu,
            out_root=tmp / "gpu_kappa_only",
            seed=args.seed,
            dim=args.dimension,
            disable_cuda=False,
            extra_args=["-e", "-r", str(ref_inf_coord), "-b", str(beta_for_kappa_only)],
            expected_return_codes=(13,),
        )

        cpu_kappa_only = parse_kappas(edge_cpu.with_suffix(".kappas"))
        gpu_kappa_only = parse_kappas(edge_gpu.with_suffix(".kappas"))
        max_kappa_diff_only = compare_list_kappas(cpu_kappa_only, gpu_kappa_only, args.kappa_tol)

        print("CPU/GPU kappa-beta A/B checks passed")
        print(f"  beta diff (-p run): {beta_diff:.6g}")
        print(f"  max kappa diff (-p run): {max_kappa_diff_post:.6g}")
        print(f"  max kappa diff (-e run): {max_kappa_diff_only:.6g}")


if __name__ == "__main__":
    main()
