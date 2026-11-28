#!/usr/bin/env bash
set -euo pipefail
echo "[098_jlens_hybrid_w8_abl_emeans_snr_70m] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-hyb_emeans_snr}"
PYTHONPATH=src python3 src/train.py \
  --linear-mode bf16 --jlens-mode hybrid \
  --hybrid-w8-weight 0.06 --hybrid-w8-start-frac 0.0 --hybrid-w8-end-frac 0.55 --hybrid-w8-every 4 \
  --jlens-weight 0.50 --jlens-start-frac 0.20 --jlens-end-frac 0.75 --jlens-every 2 --jlens-upd-every 25 \
  --abl-enable-early 0 --abl-anchor-layer 9 --abl-active-frac 0.005 \
  --abl-pool-size 64 --abl-k 8 --abl-window 1 --abl-select-mode emeans \
  --abl-loss-kind exp_signed_clamped --abl-exp-beta 6.0 \
  --abl-gamma-max 1.0 --abl-bystander-tau 0.05 --abl-bystander-w 0.25 \
  --abl-live-target-head 1 --abl-nll-cap 6.5 \
  --opt muon --muon-lr 0.025 --adam-lr 0.008 --wd 0.01 \
  --micro-batch 256 --batch-tokens 262144 --steps 2000 --seed 42 --compile 1
