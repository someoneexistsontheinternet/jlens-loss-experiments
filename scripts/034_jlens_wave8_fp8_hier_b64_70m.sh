#!/usr/bin/env bash
set -euo pipefail
echo "[034_jlens_wave8_fp8_hier_b64_70m] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-jlens_fp8_hier_b64}"
PYTHONPATH=src python3 src/train.py --linear-mode fp8_hier_bf16_fp32 --fp8-block-k 64 --hier-w-2d 0 --jlens-mode wave8 --jlens-weight 0.35 --opt muon --muon-lr 0.025 --adam-lr 0.008 --wd 0.01 --micro-batch 256 --batch-tokens 262144 --steps 2000 --compile 1
