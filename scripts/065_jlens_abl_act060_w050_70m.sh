#!/usr/bin/env bash
set -euo pipefail
echo "[065_jlens_abl_act060_w050_70m] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-jlens_abl_act060_w050}"
PYTHONPATH=src python3 src/train.py \
  --linear-mode bf16 \
  --jlens-mode ablation \
  --jlens-weight 0.50 \
  --jlens-start-frac 0.20 \
  --jlens-end-frac 0.75 \
  --jlens-every 2 \
  --abl-layer-mode fixed \
  --abl-active-frac 0.060 \
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
