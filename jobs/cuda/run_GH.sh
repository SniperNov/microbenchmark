#!/usr/bin/env bash
# GH200: original baseline, external CUDA timeline, separate UM diagnostics.
set -euo pipefail
if [ -n "${SLURM_JOB_ID:-}" ]; then
    cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
else
    cd "$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
fi
export CUDA_MACHINE=GH200
export CUDA_TRACE_UM="${CUDA_TRACE_UM:-1}"
exec bash jobs/cuda/run_suite.sh
