#!/usr/bin/env bash
set -euo pipefail
echo "[029_mxfp8_b128_ceil240_70m] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-mxfp8_b128_ceil240}"
PYTHONPATH=src python3 src/train.py --linear-mode mxfp8_e8m0 --fp8-block-k 128 --e8m0-mode ceil_240 --opt muon --muon-lr 0.025 --adam-lr 0.008 --wd 0.01 --micro-batch 256 --batch-tokens 262144 --steps 2000 --compile 1
