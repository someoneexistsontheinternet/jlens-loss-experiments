#!/usr/bin/env bash
set -euo pipefail
echo "[018_bf16_muon_bs524k_70m] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-bf16_muon_bs524k}"
PYTHONPATH=src python3 src/train.py --linear-mode bf16 --opt muon --muon-lr 0.03 --adam-lr 0.0045 --wd 0.01 --micro-batch 256 --batch-tokens 524288 --steps 1000 --warmup 50 --eval-every 125 --compile 1
