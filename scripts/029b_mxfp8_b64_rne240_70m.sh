#!/usr/bin/env bash
set -euo pipefail
echo "[029b_mxfp8_b64_rne240_70m] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-mxfp8_b64_rne240}"
PYTHONPATH=src python3 src/train.py --linear-mode mxfp8_e8m0 --fp8-block-k 64 --e8m0-mode rne_240 --opt muon --muon-lr 0.025 --adam-lr 0.008 --wd 0.01 --micro-batch 256 --batch-tokens 262144 --steps 2000 --compile 1
