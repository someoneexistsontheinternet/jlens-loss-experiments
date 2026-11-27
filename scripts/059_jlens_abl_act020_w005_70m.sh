#!/usr/bin/env bash
set -euo pipefail
echo "[059_jlens_abl_act020_w005_70m] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-jlens_abl_act020_w005}"
PYTHONPATH=src python3 src/train.py \
  --linear-mode bf16 \
  --jlens-mode ablation \
  --jlens-weight 0.05 \
  --jlens-start-frac 0.20 \
  --jlens-end-frac 0.75 \
  --jlens-every 2 \
  --abl-active-frac 0.020 \
  --abl-pool-size 64 \
  --abl-k 8 \
  --abl-window 4 \
  --abl-metric cov \
  --abl-loss-kind superpos_unlikelihood \
  --opt muon \
  --muon-lr 0.025 \
  --adam-lr 0.008 \
  --wd 0.01 \
  --micro-batch 256 \
  --batch-tokens 262144 \
  --steps 2000 \
  --compile 1
