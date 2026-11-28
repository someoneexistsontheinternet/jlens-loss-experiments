#!/usr/bin/env bash
set -euo pipefail
echo "[091_jlens_abl_select_maxmin_70m] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-abl_maxmin}"
PYTHONPATH=src python3 tools/verify_jlens_ablation.py
PYTHONPATH=src python3 src/train.py \
  --linear-mode bf16 --jlens-mode ablation \
  --jlens-weight 0.50 --jlens-start-frac 0.20 --jlens-end-frac 0.75 --jlens-every 2 --jlens-upd-every 25 \
  --abl-enable-early 0 --abl-anchor-layer 9 --abl-active-frac 0.005 \
  --abl-pool-size 64 --abl-k 8 --abl-window 1 --abl-select-mode maxmin \
  --abl-loss-kind exp_signed_clamped --abl-exp-beta 6.0 \
  --opt muon --muon-lr 0.025 --adam-lr 0.008 --wd 0.01 \
  --micro-batch 256 --batch-tokens 262144 --steps 2000 --seed 42 --compile 1
