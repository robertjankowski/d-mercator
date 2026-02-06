#!/usr/bin/env python3

import argparse
import math
import pathlib
import random
import subprocess
from typing import Iterable, List


DEFAULT_SIZES = [1000, 5000]#, 10000, 20000, 50000, 100000]


def parse_sizes(raw: str) -> List[int]:
    sizes = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not sizes:
        raise argparse.ArgumentTypeError("At least one size must be provided.")
    for size in sizes:
        if size <= 1:
            raise argparse.ArgumentTypeError("All sizes must be > 1.")
    return sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate S^1 benchmark edgelists using generatingSD_unix.cpp "
            "(gamma=2.7, <k>=10 by default) for embedding performance analysis."
        )
    )
    parser.add_argument(
        "--output-folder",
        type=pathlib.Path,
        required=True,
        help="Destination folder for generated hidden variables and edgelists.",
    )
    parser.add_argument(
        "--sizes",
        type=parse_sizes,
        default=DEFAULT_SIZES,
        help="Comma-separated network sizes. Default: 1000,5000,10000,20000,50000,100000",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.7,
        help="Power-law exponent gamma for kappa sampling. Default: 2.7",
    )
    parser.add_argument(
        "--mean-degree",
        type=float,
        default=10.0,
        help="Target mean degree <k>. Default: 10.0",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=2.5,
        help=(
            "S^1 beta parameter passed to generatingSD (required by generatingSD). "
            "Default: 2.5"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Base seed used for kappa/theta generation. Default: 12345",
    )
    parser.add_argument(
        "--generator-bin",
        type=pathlib.Path,
        default=None,
        help=(
            "Path to an existing generatingSD binary. If omitted, the script compiles "
            "src/generatingSD_unix.cpp into ./gen_net_s1_bench first."
        ),
    )
    parser.add_argument(
        "--remove-hidden",
        action="store_true",
        help="Remove hidden variables files after edge generation.",
    )
    return parser.parse_args()


def repo_root_from_script() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def generate_kappas(n: int, gamma: float, mean_degree: float, rng: random.Random) -> List[float]:
    # Same sampling logic as test/generate_synthetic_networks.py (derived from generatingSD workflow).
    kappa_0 = (
        (1.0 - 1.0 / n)
        / (1.0 - n ** ((2.0 - gamma) / (gamma - 1.0)))
        * (gamma - 2.0)
        / (gamma - 1.0)
        * mean_degree
    )
    kappa_c = kappa_0 * n ** (1.0 / (gamma - 1.0))
    kappas = []
    coeff = 1.0 - (kappa_c / kappa_0) ** (1.0 - gamma)
    exponent = 1.0 / (1.0 - gamma)
    for _ in range(n):
        u = rng.random()
        value = kappa_0 * (1.0 - u * coeff) ** exponent
        # Guard tiny floating errors.
        if not math.isfinite(value) or value <= 0:
            value = kappa_0
        kappas.append(value)
    return kappas


def write_hidden_variables_file(path: pathlib.Path, kappas: Iterable[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in kappas:
            handle.write(f"{value:.12f}\n")


def ensure_generator_binary(repo_root: pathlib.Path, requested: pathlib.Path | None) -> pathlib.Path:
    if requested is not None:
        binary = requested.resolve()
        if not binary.exists():
            raise FileNotFoundError(f"Generator binary does not exist: {binary}")
        return binary

    binary = repo_root / "gen_net_s1_bench"
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


def generate_edgelist(
    generator_bin: pathlib.Path,
    hidden_file: pathlib.Path,
    output_root: pathlib.Path,
    beta: float,
    seed: int,
    repo_root: pathlib.Path,
) -> pathlib.Path:
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
        str(hidden_file),
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)
    return output_root.with_suffix(".edge")


def main() -> None:
    args = parse_args()
    repo_root = repo_root_from_script()
    output_folder = args.output_folder.resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    generator_bin = ensure_generator_binary(repo_root, args.generator_bin)
    rng = random.Random(args.seed)

    print(f"Generator binary: {generator_bin}")
    print(f"Output folder: {output_folder}")
    print(
        f"Model parameters: D=1, gamma={args.gamma}, <k>={args.mean_degree}, "
        f"beta={args.beta}, sizes={args.sizes}"
    )

    for idx, n in enumerate(args.sizes):
        case_folder = output_folder / f"N_{n}"
        case_folder.mkdir(parents=True, exist_ok=True)

        kappas = generate_kappas(n, args.gamma, args.mean_degree, rng)
        hidden_file = case_folder / f"s1_hidden_N{n}.txt"
        output_root = case_folder / f"s1_gamma{str(args.gamma).replace('.', '_')}_k{str(args.mean_degree).replace('.', '_')}_N{n}"
        seed = args.seed + idx

        write_hidden_variables_file(hidden_file, kappas)
        edge_path = generate_edgelist(
            generator_bin=generator_bin,
            hidden_file=hidden_file,
            output_root=output_root,
            beta=args.beta,
            seed=seed,
            repo_root=repo_root,
        )
        print(f"Generated: {edge_path}")

        if args.remove_hidden:
            hidden_file.unlink(missing_ok=True)

    print("Done.")


if __name__ == "__main__":
    main()
