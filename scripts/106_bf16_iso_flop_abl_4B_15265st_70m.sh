#!/usr/bin/env bash
set -euo pipefail
echo "[106_bf16_iso_flop_abl_4B_15265st_70m] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-bf16_iso_abl_4B}"
PYTHONPATH=src python3 src/train.py \
  --linear-mode bf16 --jlens-mode none \
  --opt muon --muon-lr 0.025 --adam-lr 0.008 --wd 0.01 \
  --micro-batch 256 --batch-tokens 262144 \
  --steps 15265 --warmup 300 --eval-every 500 --log-every 50 \
  --seed 42 --compile 1
