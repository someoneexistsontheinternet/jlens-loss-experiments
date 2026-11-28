#!/usr/bin/env bash
set -euo pipefail
echo "[158_bf16_iso_flop_2015st_seed137_70m] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-bf16_iso_flop_2015st_s137}"
PYTHONPATH=src python3 src/train.py \
  --linear-mode bf16 \
  --opt muon \
  --muon-lr 0.025 \
  --adam-lr 0.008 \
  --wd 0.01 \
  --micro-batch 256 \
  --batch-tokens 262144 \
  --steps 2015 \
  --seed 137 \
  --compile 1
