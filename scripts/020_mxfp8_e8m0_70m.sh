#!/usr/bin/env bash
set -euo pipefail
echo "[020_mxfp8_e8m0_70m] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-mxfp8_e8m0}"
PYTHONPATH=src python3 src/train.py   --linear-mode mxfp8_e8m0   --fp8-block-k 64   --opt muon   --muon-lr 0.02   --adam-lr 0.003   --wd 0.01   --micro-batch 128   --batch-tokens 262144   --steps 2000   --compile 0
