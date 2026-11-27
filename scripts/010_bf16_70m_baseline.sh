#!/usr/bin/env bash
set -euo pipefail
echo "[010_bf16_70m_baseline] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-bf16_base}"
PYTHONPATH=src python3 src/train.py   --linear-mode bf16   --opt muon   --muon-lr 0.02   --adam-lr 0.003   --wd 0.01   --micro-batch 128   --batch-tokens 262144   --steps 2000   --compile 1
