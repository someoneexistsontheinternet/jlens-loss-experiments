#!/usr/bin/env bash
set -euo pipefail
echo "[014_bf16_adamw_lr15e4_70m] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-bf16_adamw_lr15e4}"
PYTHONPATH=src python3 src/train.py --linear-mode bf16 --opt adamw --adam-lr 0.0015 --wd 0.05 --micro-batch 256 --batch-tokens 262144 --steps 2000 --compile 1
