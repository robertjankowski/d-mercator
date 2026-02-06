#!/usr/bin/env python3

import argparse
import csv
import json
import os
import pathlib
import subprocess
import tempfile
import time
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Embed all edgelists in a folder with CPU and GPU paths, "
            "measure wall-clock time, and report per-file comparison."
        )
    )
    parser.add_argument(
        "--input-folder",
        type=pathlib.Path,
        required=True,
        help="Folder containing edgelists (*.edge by default).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.edge",
        help="Glob pattern for edgelists. Default: *.edge",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search edgelists recursively in the input folder.",
    )
    parser.add_argument(
        "--cpu-bin",
        type=pathlib.Path,
        required=True,
        help="Path to CPU embedding binary (mercator).",
    )
    parser.add_argument(
        "--gpu-bin",
        type=pathlib.Path,
        required=True,
        help="Path to GPU embedding binary (mercator built with CUDA).",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=1,
        help="Embedding dimension passed to mercator (-d). Default: 1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Seed passed to mercator (-s). Default: 12345",
    )
    parser.add_argument(
        "--output-json",
        type=pathlib.Path,
        default=None,
        help="Optional JSON output path with benchmark results.",
    )
    parser.add_argument(
        "--output-csv",
        type=pathlib.Path,
        default=None,
        help="Optional CSV output path with benchmark results.",
    )
    parser.add_argument(
        "--omp-threads",
        type=int,
        default=None,
        help="Optional OMP_NUM_THREADS for both runs.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argument appended to mercator command (can be repeated).",
    )
    return parser.parse_args()


def find_edgelists(folder: pathlib.Path, pattern: str, recursive: bool) -> List[pathlib.Path]:
    if recursive:
        files = sorted([path for path in folder.rglob(pattern) if path.is_file()])
    else:
        files = sorted([path for path in folder.glob(pattern) if path.is_file()])
    return files


def run_embedding(
    binary: pathlib.Path,
    edge_file: pathlib.Path,
    output_root: pathlib.Path,
    dimension: int,
    seed: int,
    extra_args: List[str],
    disable_cuda: bool,
    omp_threads: int | None,
) -> Dict[str, object]:
    cmd = [
        str(binary),
        "-q",
        "-d",
        str(dimension),
        "-s",
        str(seed),
        "-o",
        str(output_root),
        *extra_args,
        str(edge_file),
    ]

    env = os.environ.copy()
    if disable_cuda:
        env["DMERCATOR_DISABLE_CUDA"] = "1"
    else:
        env.pop("DMERCATOR_DISABLE_CUDA", None)
    if omp_threads is not None:
        env["OMP_NUM_THREADS"] = str(omp_threads)
        env["OMP_DYNAMIC"] = "FALSE"

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    elapsed = time.perf_counter() - t0

    return {
        "elapsed_sec": elapsed,
        "return_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "command": cmd,
    }


def print_table(rows: List[Dict[str, object]]) -> None:
    header = f"{'edgelist':60} {'cpu_sec':>12} {'gpu_sec':>12} {'speedup':>10} {'status':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        cpu = row["cpu_sec"]
        gpu = row["gpu_sec"]
        status = row["status"]
        speedup = (cpu / gpu) if (status == "ok" and gpu > 0) else float("nan")
        speedup_str = f"{speedup:.3f}" if status == "ok" else "n/a"
        print(
            f"{row['edgelist']:60} "
            f"{cpu:12.4f} "
            f"{gpu:12.4f} "
            f"{speedup_str:>10} "
            f"{status:>10}"
        )


def write_outputs(rows: List[Dict[str, object]], json_path: pathlib.Path | None, csv_path: pathlib.Path | None) -> None:
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        print(f"\nJSON written to: {json_path}")

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["edgelist", "cpu_sec", "gpu_sec", "speedup", "status", "cpu_rc", "gpu_rc"],
            )
            writer.writeheader()
            for row in rows:
                speedup = (row["cpu_sec"] / row["gpu_sec"]) if (row["status"] == "ok" and row["gpu_sec"] > 0) else None
                writer.writerow(
                    {
                        "edgelist": row["edgelist"],
                        "cpu_sec": f"{row['cpu_sec']:.6f}",
                        "gpu_sec": f"{row['gpu_sec']:.6f}",
                        "speedup": f"{speedup:.6f}" if speedup is not None else "",
                        "status": row["status"],
                        "cpu_rc": row["cpu_rc"],
                        "gpu_rc": row["gpu_rc"],
                    }
                )
        print(f"CSV written to: {csv_path}")


def main() -> None:
    args = parse_args()

    input_folder = args.input_folder.resolve()
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
    if not args.cpu_bin.resolve().exists():
        raise FileNotFoundError(f"CPU binary does not exist: {args.cpu_bin.resolve()}")
    if not args.gpu_bin.resolve().exists():
        raise FileNotFoundError(f"GPU binary does not exist: {args.gpu_bin.resolve()}")

    edgelists = find_edgelists(input_folder, args.pattern, args.recursive)
    if not edgelists:
        raise RuntimeError(f"No edgelists found in {input_folder} with pattern '{args.pattern}'")

    print(f"Found {len(edgelists)} edgelist(s).")

    rows: List[Dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="dmercator_embed_bench_") as tmp_dir:
        tmp_root = pathlib.Path(tmp_dir)

        for idx, edge_file in enumerate(edgelists):
            cpu_out = tmp_root / f"cpu_{idx}"
            gpu_out = tmp_root / f"gpu_{idx}"

            cpu_result = run_embedding(
                binary=args.cpu_bin.resolve(),
                edge_file=edge_file,
                output_root=cpu_out,
                dimension=args.dimension,
                seed=args.seed,
                extra_args=args.extra_arg,
                disable_cuda=True,
                omp_threads=args.omp_threads,
            )
            gpu_result = run_embedding(
                binary=args.gpu_bin.resolve(),
                edge_file=edge_file,
                output_root=gpu_out,
                dimension=args.dimension,
                seed=args.seed,
                extra_args=args.extra_arg,
                disable_cuda=False,
                omp_threads=args.omp_threads,
            )

            status = "ok" if (cpu_result["return_code"] == 0 and gpu_result["return_code"] == 0) else "failed"
            row = {
                "edgelist": str(edge_file),
                "cpu_sec": float(cpu_result["elapsed_sec"]),
                "gpu_sec": float(gpu_result["elapsed_sec"]),
                "cpu_rc": int(cpu_result["return_code"]),
                "gpu_rc": int(gpu_result["return_code"]),
                "status": status,
                "cpu_cmd": cpu_result["command"],
                "gpu_cmd": gpu_result["command"],
                "cpu_stderr": cpu_result["stderr"],
                "gpu_stderr": gpu_result["stderr"],
            }
            rows.append(row)

    print_table(rows)
    write_outputs(rows, args.output_json, args.output_csv)

    n_ok = sum(1 for row in rows if row["status"] == "ok")
    n_fail = len(rows) - n_ok
    print(f"\nCompleted. ok={n_ok}, failed={n_fail}")


if __name__ == "__main__":
    main()
