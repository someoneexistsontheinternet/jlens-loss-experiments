#!/usr/bin/env bash
set -euo pipefail
echo "[021_fp8_hier_bf16_fp32_70m] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-fp8_hier}"
PYTHONPATH=src python3 src/train.py   --linear-mode fp8_hier_bf16_fp32   --fp8-block-k 128   --opt muon   --muon-lr 0.02   --adam-lr 0.003   --wd 0.01   --micro-batch 128   --batch-tokens 262144   --steps 2000   --compile 0
