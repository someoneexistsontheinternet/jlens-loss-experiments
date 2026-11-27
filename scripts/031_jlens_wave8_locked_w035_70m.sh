#!/usr/bin/env bash
set -euo pipefail
echo "[031_jlens_wave8_locked_w035_70m] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-jlens_bf16_w035}"
PYTHONPATH=src python3 src/train.py --linear-mode bf16 --jlens-mode wave8 --jlens-weight 0.35 --opt muon --muon-lr 0.025 --adam-lr 0.008 --wd 0.01 --micro-batch 256 --batch-tokens 262144 --steps 2000 --compile 1
