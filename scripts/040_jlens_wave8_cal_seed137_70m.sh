#!/usr/bin/env bash
set -euo pipefail
echo "[040_jlens_wave8_cal_seed137_70m] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-jlens_wave8_seed137}"
PYTHONPATH=src python3 src/train.py --linear-mode bf16 --jlens-mode wave8 --jlens-horizons 8 --jlens-weight 0.05 --jlens-end-frac 0.75 --opt muon --muon-lr 0.025 --adam-lr 0.008 --wd 0.01 --micro-batch 256 --batch-tokens 262144 --steps 2000 --seed 137 --compile 1
