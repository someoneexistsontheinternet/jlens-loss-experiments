#!/usr/bin/env bash
set -euo pipefail
echo "[105_bf16_4B_15200st_70m] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-bf16_4B_15200}"
PYTHONPATH=src python3 src/train.py \
  --linear-mode bf16 --jlens-mode none \
  --opt muon --muon-lr 0.025 --adam-lr 0.008 --wd 0.01 \
  --micro-batch 256 --batch-tokens 262144 \
  --steps 15200 --warmup 300 --eval-every 500 --log-every 50 \
  --seed 42 --compile 1
