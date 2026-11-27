#!/usr/bin/env bash
set -euo pipefail
echo "[026_fp8_hier_b128_1d_70m] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-fp8_hier_b128_1d}"
PYTHONPATH=src python3 src/train.py --linear-mode fp8_hier_bf16_fp32 --fp8-block-k 128 --hier-w-2d 0 --opt muon --muon-lr 0.025 --adam-lr 0.008 --wd 0.01 --micro-batch 256 --batch-tokens 262144 --steps 2000 --compile 1
