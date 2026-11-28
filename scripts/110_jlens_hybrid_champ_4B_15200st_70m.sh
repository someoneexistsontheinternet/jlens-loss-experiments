#!/usr/bin/env bash
set -euo pipefail
echo "[110_jlens_hybrid_champ_4B_15200st_70m] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-hyb_champ_4B}"
PYTHONPATH=src python3 src/train.py \
  --linear-mode bf16 --jlens-mode hybrid \
  --hybrid-w8-layers "3,5,7,9" --hybrid-w8-weight 0.05 --hybrid-w8-start-frac 0.0 --hybrid-w8-end-frac 0.50 --hybrid-w8-every 2 \
  --jlens-weight 0.50 --jlens-start-frac 0.20 --jlens-end-frac 0.75 --jlens-every 2 --jlens-upd-every 25 \
  --abl-enable-early 0 --abl-anchor-layer 9 --abl-active-frac 0.005 \
  --abl-pool-size 256 --abl-k 8 --abl-window 1 --abl-select-mode kmeans \
  --abl-loss-kind exp_signed_clamped --abl-exp-beta 6.0 \
  --abl-gamma-max 1.0 --abl-bystander-tau 0.05 --abl-bystander-w 0.25 \
  --opt muon --muon-lr 0.025 --adam-lr 0.008 --wd 0.01 \
  --micro-batch 256 --batch-tokens 262144 \
  --steps 15200 --warmup 300 --eval-every 500 --log-every 50 \
  --seed 42 --compile 1
