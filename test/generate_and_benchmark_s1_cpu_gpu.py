#!/usr/bin/env python3

import argparse
import csv
import math
import os
import pathlib
import random
import subprocess
import time
from typing import Any, Dict, Iterable, List


DEFAULT_SIZES = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]


def parse_size_token(token: str) -> int:
    value = token.strip().lower().replace("_", "")
    if not value:
        raise argparse.ArgumentTypeError("Size token cannot be empty.")
    if value.endswith("k"):
        base = value[:-1]
        if not base:
            raise argparse.ArgumentTypeError(f"Invalid size token '{token}'.")
        return int(float(base) * 1000)
    return int(value)


def parse_sizes(raw: str) -> List[int]:
    sizes = [parse_size_token(token) for token in raw.split(",") if token.strip()]
    if not sizes:
        raise argparse.ArgumentTypeError("At least one network size must be provided.")
    for size in sizes:
        if size <= 1:
            raise argparse.ArgumentTypeError("All sizes must be > 1.")
    return sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate S^1 benchmark networks with generatingSD_unix (D=1), "
            "embed each network with CPU and GPU modes, and write a timing CSV."
        )
    )
    parser.add_argument(
        "--output-folder",
        type=pathlib.Path,
        required=True,
        help="Root output folder for generated networks, embeddings, and CSV.",
    )
    parser.add_argument(
        "--sizes",
        type=parse_sizes,
        default=DEFAULT_SIZES,
        help="Comma-separated sizes. Supports 'k' suffix (e.g., 10k).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.7,
        help="Power-law exponent gamma used to sample kappas. Default: 2.7",
    )
    parser.add_argument(
        "--mean-degree",
        type=float,
        default=10.0,
        help="Target mean degree <k> used for kappa sampling. Default: 10.0",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=2.5,
        help="S^1 beta parameter passed to generatingSD. Default: 2.5",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Base random seed for generation and embedding. Default: 12345",
    )
    parser.add_argument(
        "--generator-bin",
        type=pathlib.Path,
        default=None,
        help=(
            "Path to an existing generatingSD binary. If omitted, the script compiles "
            "src/generatingSD_unix.cpp to ./gen_net_s1_bench."
        ),
    )
    parser.add_argument(
        "--gpu-bin",
        type=pathlib.Path,
        required=True,
        help="Path to mercator binary used for GPU-mode embedding.",
    )
    parser.add_argument(
        "--cpu-bin",
        type=pathlib.Path,
        default=None,
        help=(
            "Optional path to mercator binary used for CPU embedding. "
            "If omitted, --gpu-bin is reused with DMERCATOR_DISABLE_CUDA=1."
        ),
    )
    parser.add_argument(
        "--omp-threads",
        type=int,
        default=None,
        help="Optional OMP_NUM_THREADS value for both CPU and GPU runs.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argument appended to each mercator command (repeatable).",
    )
    parser.add_argument(
        "--reuse-existing-edgelists",
        action="store_true",
        help="Skip generation when the expected .edge file already exists.",
    )
    parser.add_argument(
        "--keep-hidden",
        action="store_true",
        help="Keep hidden-variable files. By default they are removed after generation.",
    )
    parser.add_argument(
        "--csv-path",
        type=pathlib.Path,
        default=None,
        help=(
            "Optional CSV output path. Default: "
            "<output-folder>/cpu_gpu_timing_s1_gamma2_7_k10.csv"
        ),
    )
    return parser.parse_args()


def repo_root_from_script() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def fmt_float_for_path(value: float) -> str:
    return f"{value:g}".replace(".", "_")


def generate_kappas(n: int, gamma: float, mean_degree: float, rng: random.Random) -> List[float]:
    # Same sampling logic used in existing test/generate_synthetic_networks.py.
    kappa_0 = (
        (1.0 - 1.0 / n)
        / (1.0 - n ** ((2.0 - gamma) / (gamma - 1.0)))
        * (gamma - 2.0)
        / (gamma - 1.0)
        * mean_degree
    )
    kappa_c = kappa_0 * n ** (1.0 / (gamma - 1.0))
    coeff = 1.0 - (kappa_c / kappa_0) ** (1.0 - gamma)
    exponent = 1.0 / (1.0 - gamma)

    kappas: List[float] = []
    for _ in range(n):
        u = rng.random()
        value = kappa_0 * (1.0 - u * coeff) ** exponent
        if not math.isfinite(value) or value <= 0.0:
            value = kappa_0
        kappas.append(value)
    return kappas


def write_hidden_variables(path: pathlib.Path, kappas: Iterable[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in kappas:
            handle.write(f"{value:.12f}\n")


def run_timed_command(
    cmd: List[str],
    env: Dict[str, str] | None = None,
    cwd: pathlib.Path | None = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    elapsed = time.perf_counter() - t0
    return {
        "elapsed_sec": elapsed,
        "return_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "command": cmd,
    }


def ensure_generator_binary(repo_root: pathlib.Path, requested: pathlib.Path | None) -> pathlib.Path:
    if requested is not None:
        binary = requested.resolve()
        if not binary.exists():
            raise FileNotFoundError(f"Generator binary does not exist: {binary}")
        return binary

    binary = (repo_root / "gen_net_s1_bench").resolve()
    compile_cmd = [
        "g++",
        "-O3",
        "--std=c++17",
        "-o",
        str(binary),
        "src/generatingSD_unix.cpp",
    ]
    subprocess.run(compile_cmd, cwd=repo_root, check=True)
    return binary


def generate_network_for_size(
    generator_bin: pathlib.Path,
    repo_root: pathlib.Path,
    case_dir: pathlib.Path,
    output_root: pathlib.Path,
    n: int,
    gamma: float,
    mean_degree: float,
    beta: float,
    seed: int,
    rng: random.Random,
    reuse_existing_edgelists: bool,
    keep_hidden: bool,
) -> Dict[str, Any]:
    edge_path = output_root.with_suffix(".edge")
    hidden_path = case_dir / f"s1_hidden_N{n}.txt"

    if reuse_existing_edgelists and edge_path.exists():
        return {
            "edge_path": edge_path,
            "hidden_path": hidden_path,
            "elapsed_sec": 0.0,
            "return_code": 0,
            "stdout": "",
            "stderr": "",
            "command": [],
            "skipped": True,
        }

    kappas = generate_kappas(n=n, gamma=gamma, mean_degree=mean_degree, rng=rng)
    write_hidden_variables(hidden_path, kappas)

    cmd = [
        str(generator_bin),
        "-b",
        str(beta),
        "-d",
        "1",
        "-v",
        "-s",
        str(seed),
        "-o",
        str(output_root),
        str(hidden_path),
    ]
    result = run_timed_command(cmd=cmd, cwd=repo_root)
    result["edge_path"] = edge_path
    result["hidden_path"] = hidden_path
    result["skipped"] = False

    if result["return_code"] == 0 and not keep_hidden:
        hidden_path.unlink(missing_ok=True)

    return result


def run_embedding(
    binary: pathlib.Path,
    edge_path: pathlib.Path,
    output_root: pathlib.Path,
    seed: int,
    extra_args: List[str],
    disable_cuda: bool,
    omp_threads: int | None,
    repo_root: pathlib.Path,
) -> Dict[str, Any]:
    cmd = [
        str(binary),
        "-q",
        "-d",
        "1",
        "-s",
        str(seed),
        "-o",
        str(output_root),
        *extra_args,
        str(edge_path),
    ]
    env = os.environ.copy()
    if disable_cuda:
        env["DMERCATOR_DISABLE_CUDA"] = "1"
    else:
        env.pop("DMERCATOR_DISABLE_CUDA", None)
    if omp_threads is not None:
        env["OMP_NUM_THREADS"] = str(omp_threads)
        env["OMP_DYNAMIC"] = "FALSE"
    return run_timed_command(cmd=cmd, env=env, cwd=repo_root)


def csv_output_path(output_folder: pathlib.Path, gamma: float, mean_degree: float, custom: pathlib.Path | None) -> pathlib.Path:
    if custom is not None:
        return custom.resolve()
    default_name = f"cpu_gpu_timing_s1_gamma{fmt_float_for_path(gamma)}_k{fmt_float_for_path(mean_degree)}.csv"
    return (output_folder / default_name).resolve()


def write_csv(rows: List[Dict[str, Any]], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "requested_size",
        "edge_path",
        "generation_sec",
        "cpu_sec",
        "gpu_sec",
        "speedup_cpu_over_gpu",
        "generation_rc",
        "cpu_rc",
        "gpu_rc",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            speedup = ""
            if row["status"] == "ok" and row["gpu_sec"] > 0:
                speedup = f"{row['cpu_sec'] / row['gpu_sec']:.6f}"
            writer.writerow(
                {
                    "requested_size": row["requested_size"],
                    "edge_path": row["edge_path"],
                    "generation_sec": f"{row['generation_sec']:.6f}",
                    "cpu_sec": f"{row['cpu_sec']:.6f}",
                    "gpu_sec": f"{row['gpu_sec']:.6f}",
                    "speedup_cpu_over_gpu": speedup,
                    "generation_rc": row["generation_rc"],
                    "cpu_rc": row["cpu_rc"],
                    "gpu_rc": row["gpu_rc"],
                    "status": row["status"],
                }
            )


def print_summary(rows: List[Dict[str, Any]], csv_path: pathlib.Path) -> None:
    header = (
        f"{'N':>9} {'gen_sec':>12} {'cpu_sec':>12} {'gpu_sec':>12} "
        f"{'speedup':>10} {'status':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        speedup = row["cpu_sec"] / row["gpu_sec"] if (row["status"] == "ok" and row["gpu_sec"] > 0) else float("nan")
        speedup_str = f"{speedup:.3f}" if row["status"] == "ok" else "n/a"
        print(
            f"{row['requested_size']:9d} "
            f"{row['generation_sec']:12.4f} "
            f"{row['cpu_sec']:12.4f} "
            f"{row['gpu_sec']:12.4f} "
            f"{speedup_str:>10} "
            f"{row['status']:>12}"
        )
    ok_count = sum(1 for row in rows if row["status"] == "ok")
    print(f"\nCompleted {len(rows)} case(s): ok={ok_count}, failed={len(rows) - ok_count}")
    print(f"CSV written to: {csv_path}")


def main() -> None:
    args = parse_args()
    repo_root = repo_root_from_script()

    output_folder = args.output_folder.resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    networks_folder = output_folder / "networks"
    embeddings_folder = output_folder / "embeddings"
    csv_path = csv_output_path(output_folder, args.gamma, args.mean_degree, args.csv_path)

    generator_bin = ensure_generator_binary(repo_root=repo_root, requested=args.generator_bin)
    gpu_bin = args.gpu_bin.resolve()
    cpu_bin = args.cpu_bin.resolve() if args.cpu_bin is not None else gpu_bin

    if not gpu_bin.exists():
        raise FileNotFoundError(f"GPU binary does not exist: {gpu_bin}")
    if not cpu_bin.exists():
        raise FileNotFoundError(f"CPU binary does not exist: {cpu_bin}")

    print(f"Generator: {generator_bin}")
    print(f"CPU binary: {cpu_bin}")
    print(f"GPU binary: {gpu_bin}")
    print(
        f"Generation params: D=1, gamma={args.gamma}, <k>={args.mean_degree}, "
        f"beta={args.beta}, sizes={args.sizes}"
    )

    rng = random.Random(args.seed)
    rows: List[Dict[str, Any]] = []

    for idx, n in enumerate(args.sizes):
        case_seed = args.seed + idx
        case_dir = networks_folder / f"N_{n}"
        case_dir.mkdir(parents=True, exist_ok=True)

        root_label = (
            f"s1_gamma{fmt_float_for_path(args.gamma)}_"
            f"k{fmt_float_for_path(args.mean_degree)}_N{n}"
        )
        generated_root = case_dir / root_label

        print(f"\n[{idx + 1}/{len(args.sizes)}] N={n}")
        generation = generate_network_for_size(
            generator_bin=generator_bin,
            repo_root=repo_root,
            case_dir=case_dir,
            output_root=generated_root,
            n=n,
            gamma=args.gamma,
            mean_degree=args.mean_degree,
            beta=args.beta,
            seed=case_seed,
            rng=rng,
            reuse_existing_edgelists=args.reuse_existing_edgelists,
            keep_hidden=args.keep_hidden,
        )
        edge_path = pathlib.Path(generation["edge_path"])
        if generation["skipped"]:
            print(f"  generation: skipped (reusing {edge_path})")
        else:
            print(f"  generation: {generation['elapsed_sec']:.4f}s (rc={generation['return_code']})")

        if generation["return_code"] != 0 or not edge_path.exists():
            rows.append(
                {
                    "requested_size": n,
                    "edge_path": str(edge_path),
                    "generation_sec": float(generation["elapsed_sec"]),
                    "cpu_sec": 0.0,
                    "gpu_sec": 0.0,
                    "generation_rc": int(generation["return_code"]),
                    "cpu_rc": -1,
                    "gpu_rc": -1,
                    "status": "generation_failed",
                }
            )
            print("  embedding: skipped (generation failed)")
            continue

        cpu_out = embeddings_folder / "cpu" / f"N_{n}_seed_{case_seed}"
        gpu_out = embeddings_folder / "gpu" / f"N_{n}_seed_{case_seed}"
        cpu_out.parent.mkdir(parents=True, exist_ok=True)
        gpu_out.parent.mkdir(parents=True, exist_ok=True)

        cpu_result = run_embedding(
            binary=cpu_bin,
            edge_path=edge_path,
            output_root=cpu_out,
            seed=case_seed,
            extra_args=args.extra_arg,
            disable_cuda=True,
            omp_threads=args.omp_threads,
            repo_root=repo_root,
        )
        gpu_result = run_embedding(
            binary=gpu_bin,
            edge_path=edge_path,
            output_root=gpu_out,
            seed=case_seed,
            extra_args=args.extra_arg,
            disable_cuda=False,
            omp_threads=args.omp_threads,
            repo_root=repo_root,
        )

        status = "ok" if (cpu_result["return_code"] == 0 and gpu_result["return_code"] == 0) else "embed_failed"
        rows.append(
            {
                "requested_size": n,
                "edge_path": str(edge_path),
                "generation_sec": float(generation["elapsed_sec"]),
                "cpu_sec": float(cpu_result["elapsed_sec"]),
                "gpu_sec": float(gpu_result["elapsed_sec"]),
                "generation_rc": int(generation["return_code"]),
                "cpu_rc": int(cpu_result["return_code"]),
                "gpu_rc": int(gpu_result["return_code"]),
                "status": status,
            }
        )
        print(
            f"  cpu: {cpu_result['elapsed_sec']:.4f}s (rc={cpu_result['return_code']}), "
            f"gpu: {gpu_result['elapsed_sec']:.4f}s (rc={gpu_result['return_code']})"
        )

    write_csv(rows=rows, path=csv_path)
    print_summary(rows=rows, csv_path=csv_path)


if __name__ == "__main__":
    main()
