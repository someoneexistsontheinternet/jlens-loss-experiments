#!/usr/bin/env bash
set -euo pipefail
echo "[030_jlens_wave8_70m] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-jlens_wave8}"
PYTHONPATH=src python3 src/train.py   --linear-mode bf16   --jlens-mode wave8   --jlens-weight 0.35   --opt muon   --muon-lr 0.02   --adam-lr 0.003   --wd 0.01   --micro-batch 128   --batch-tokens 262144   --steps 2000   --compile 1
