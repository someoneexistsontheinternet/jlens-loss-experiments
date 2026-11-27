#!/usr/bin/env bash
set -euo pipefail
echo "[024_fp8_rowwise_70m_locked] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-fp8_rowwise_locked}"
PYTHONPATH=src python3 src/train.py --linear-mode fp8_rowwise --opt muon --muon-lr 0.025 --adam-lr 0.008 --wd 0.01 --micro-batch 256 --batch-tokens 262144 --steps 2000 --compile 1
