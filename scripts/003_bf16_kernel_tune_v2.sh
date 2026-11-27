#!/usr/bin/env bash
set -euo pipefail
echo "[003_bf16_kernel_tune_v2] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-bf16_tune_v2}"
PYTHONPATH=src python3 src/bench_bf16_step.py --log-dir "${DATA_ROOT:-/data}/logs"
