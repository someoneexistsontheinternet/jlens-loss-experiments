#!/usr/bin/env bash
set -euo pipefail
echo "[041_jlens_wave8_cal_fp8_rowwise_70m] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-jlens_wave8_fp8_row}"
PYTHONPATH=src python3 src/train.py --linear-mode fp8_rowwise --jlens-mode wave8 --jlens-horizons 8 --jlens-weight 0.035 --jlens-end-frac 0.65 --opt muon --muon-lr 0.025 --adam-lr 0.008 --wd 0.01 --micro-batch 256 --batch-tokens 262144 --steps 2000 --seed 42 --compile 1
