#!/usr/bin/env bash
set -euo pipefail
echo "[022_fp8_rowwise_70m] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-fp8_rowwise}"
PYTHONPATH=src python3 src/train.py   --linear-mode fp8_rowwise   --opt muon   --muon-lr 0.02   --adam-lr 0.003   --wd 0.01   --micro-batch 128   --batch-tokens 262144   --steps 2000   --compile 0
