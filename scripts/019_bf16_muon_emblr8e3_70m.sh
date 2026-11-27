#!/usr/bin/env bash
set -euo pipefail
echo "[019_bf16_muon_emblr8e3_70m] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-bf16_muon_emblr8e3}"
PYTHONPATH=src python3 src/train.py --linear-mode bf16 --opt muon --muon-lr 0.025 --adam-lr 0.008 --wd 0.01 --micro-batch 256 --batch-tokens 262144 --steps 2000 --compile 1
