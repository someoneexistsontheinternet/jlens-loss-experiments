#!/usr/bin/env bash
set -euo pipefail
echo "[052_bf16_scale_2200st] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-bf16_2200st}"
PYTHONPATH=src python3 src/train.py \
  --linear-mode bf16 \
  --opt muon \
  --muon-lr 0.025 \
  --adam-lr 0.008 \
  --wd 0.01 \
  --micro-batch 256 \
  --batch-tokens 262144 \
  --steps 2200 \
  --compile 1
