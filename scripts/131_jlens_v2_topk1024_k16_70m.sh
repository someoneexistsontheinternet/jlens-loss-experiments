#!/usr/bin/env bash
set -euo pipefail
echo "[131_jlens_v2_topk1024_k16_70m] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-v2_topk1024_k16}"
PYTHONPATH=src python3 src/train.py \
  --linear-mode bf16 --jlens-mode ablation \
  --jlens-weight 0.50 --jlens-start-frac 0.20 --jlens-end-frac 0.75 --jlens-every 2 --jlens-upd-every 25 \
  --abl-enable-early 0 --abl-layer-mode split_6_9 --abl-active-frac 0.003 \
  --abl-pool-size 1024 --abl-k 16 --abl-window 1 --abl-select-mode kmeans \
  --abl-metric euclidean --abl-j-mode true_jacobian --abl-jac-samples 16 --abl-live-readout 1 --abl-reuse-clean-nll 1 \
  --abl-loss-kind exp_signed_clamped --abl-exp-beta 6.0 \
  --abl-gamma-max 1.0 --abl-bystander-tau 0.05 --abl-bystander-w 0.25 \
  --opt muon --muon-lr 0.025 --adam-lr 0.008 --wd 0.01 \
  --micro-batch 256 --batch-tokens 262144 --steps 2000 --seed 42 --compile 1
