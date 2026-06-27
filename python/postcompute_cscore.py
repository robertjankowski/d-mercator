#!/usr/bin/env python3

import argparse
import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from topology_from_embeddings import SUMMARY_FIELDS, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-compute exact S1 C-scores from existing generated and inferred coordinates."
    )
    parser.add_argument("--job-dir", required=True, help="Existing benchmark run directory")
    parser.add_argument("--cscore-binary", required=True, help="Path to compute_cscore_fast")
    parser.add_argument("--force", action="store_true", help="Ignore per-embedding C-score caches")
    return parser.parse_args()


def atomic_write_json(path: Path, payload: Any) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in SUMMARY_FIELDS})
        temporary = Path(handle.name)
    os.replace(temporary, path)


def compute_one(
    binary: Path,
    generated_coordinates: Path,
    inferred_coordinates: Path,
    cache_path: Path,
    force: bool,
) -> Dict[str, Any]:
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    completed = subprocess.run(
        [str(binary), str(generated_coordinates), str(inferred_coordinates)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    result["generated_coordinates"] = str(generated_coordinates)
    result["inferred_coordinates"] = str(inferred_coordinates)
    atomic_write_json(cache_path, result)
    return result


def main() -> int:
    args = parse_args()
    job_dir = Path(args.job_dir).resolve()
    binary = Path(args.cscore_binary).resolve()
    output_dir = job_dir / "topology_validation"
    json_path = output_dir / "topology_from_embeddings.json"
    csv_path = output_dir / "topology_from_embeddings.csv"

    if not binary.is_file():
        raise SystemExit(f"Missing C-score binary: {binary}")
    if not json_path.is_file() or not csv_path.is_file():
        raise SystemExit(f"Missing topology summary under: {output_dir}")

    records = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise SystemExit(f"No topology records found in: {json_path}")

    completed = 0
    for index, record in enumerate(records, start=1):
        if int(record["dimension"]) != 1:
            raise SystemExit("C-score post-processing currently supports only D=1 records")

        inferred_coordinates = Path(record["inf_coord_file"]).resolve()
        case_dir = inferred_coordinates.parent
        generated_coordinates = case_dir / "synthetic_sd.gen_coord"
        cache_path = Path(f"{record['root']}.cscore.json").resolve()
        if not generated_coordinates.is_file():
            raise SystemExit(f"Missing generated coordinates: {generated_coordinates}")
        if not inferred_coordinates.is_file():
            raise SystemExit(f"Missing inferred coordinates: {inferred_coordinates}")

        label = (
            f"N={record['size']} {record['backend']}/{record['sample_count']}"
        )
        print(f"[{index}/{len(records)}] Computing {label}", flush=True)
        result = compute_one(
            binary,
            generated_coordinates,
            inferred_coordinates,
            cache_path,
            args.force,
        )
        record["c_score"] = float(result["c_score"])
        completed += 1
        print(
            f"[{index}/{len(records)}] {label}: C-score={record['c_score']:.12f}",
            flush=True,
        )

    atomic_write_json(json_path, records)
    atomic_write_csv(csv_path, records)
    write_report(records, output_dir, str(job_dir))
    print(f"Updated {completed} C-scores in {csv_path}", flush=True)
    print(f"Updated topology JSON and report under {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
