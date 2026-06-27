#!/bin/bash
#SBATCH -J dmercator_cscore
#SBATCH -c 1
#SBATCH -t 240
#SBATCH -p mweber_compute
#SBATCH --mem=16000
#SBATCH --open-mode=append
#SBATCH -o cscore_%j.out
#SBATCH -e cscore_%j.err

set -euo pipefail
set -x

JOB_DIR="${JOB_DIR:-${1:-}}"
if [[ -z "${JOB_DIR}" ]]; then
  echo "Usage: sbatch run_cscore_sbatch.sh JOB_DIR" >&2
  exit 2
fi

module load gcc/12.2.0-fasrc01
module load python/3.10.12-fasrc01
source activate pt2.3.0_cuda12.1

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
BUILD_DIR="${REPO_ROOT}/build_tools"
CSCORE_BINARY="${BUILD_DIR}/compute_cscore_fast"

mkdir -p "${BUILD_DIR}"
g++ -O3 -DNDEBUG -std=c++17 \
  "${REPO_ROOT}/tools/compute_cscore_fast.cpp" \
  -o "${CSCORE_BINARY}"

"${CSCORE_BINARY}" --self-test

python3 -u "${REPO_ROOT}/python/postcompute_cscore.py" \
  --job-dir "${JOB_DIR}" \
  --cscore-binary "${CSCORE_BINARY}"
