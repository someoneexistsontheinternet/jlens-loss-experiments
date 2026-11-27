#!/usr/bin/env bash
set -euo pipefail
echo "[011_bf16_muon_lr01_70m] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-bf16_muon_lr01}"
PYTHONPATH=src python3 src/train.py --linear-mode bf16 --opt muon --muon-lr 0.01 --adam-lr 0.002 --wd 0.01 --micro-batch 256 --batch-tokens 262144 --steps 2000 --compile 1
