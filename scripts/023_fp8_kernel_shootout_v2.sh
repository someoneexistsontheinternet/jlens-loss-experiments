#!/usr/bin/env bash
set -euo pipefail
echo "[023_fp8_kernel_shootout_v2] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-fp8_bench_v2}"
PYTHONPATH=src python3 src/bench_kernels.py --bench-model 1 --log-dir "${DATA_ROOT:-/data}/logs"
