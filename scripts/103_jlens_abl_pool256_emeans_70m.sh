#!/usr/bin/env bash
set -euo pipefail
echo "[103_jlens_abl_pool256_emeans_70m] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-abl_p256_emeans}"
PYTHONPATH=src python3 src/train.py \
  --linear-mode bf16 --jlens-mode ablation \
  --jlens-weight 0.50 --jlens-start-frac 0.20 --jlens-end-frac 0.75 --jlens-every 2 --jlens-upd-every 25 \
  --abl-enable-early 0 --abl-anchor-layer 9 --abl-active-frac 0.005 \
  --abl-pool-size 256 --abl-k 8 --abl-window 1 --abl-select-mode emeans \
  --abl-loss-kind exp_signed_clamped --abl-exp-beta 6.0 \
  --opt muon --muon-lr 0.025 --adam-lr 0.008 --wd 0.01 \
  --micro-batch 256 --batch-tokens 262144 --steps 2000 --seed 42 --compile 1
