#!/usr/bin/env bash
set -euo pipefail
echo "[156_jlens_v3_6_hardcap65_70m] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-v3_6_hardcap65_s42}"
PYTHONPATH=src python3 src/train.py \
  --linear-mode bf16 --jlens-mode ablation \
  --jlens-weight 0.05 --jlens-start-step 150 --jlens-flat-end-frac 1.0 --jlens-end-frac 1.0 --jlens-every 2 --jlens-upd-every 25 \
  --abl-enable-early 0 --abl-anchor-layer 9 --abl-active-frac 0.005 \
  --abl-pool-size 64 --abl-k 8 --abl-window 1 --abl-select-mode greedy_cos \
  --abl-metric euclidean --abl-j-mode identity --abl-norm-w-dir 1 --abl-coalition-norm 0 --abl-live-readout 1 --abl-reuse-clean-nll 0 \
  --abl-pos-mode hard_easy --abl-nll-cap 6.5 --abl-loss-kind exp_unlikelihood --abl-exp-beta 6.0 \
  --abl-bystander-tau 0.0 --abl-bystander-w 0.0 --abl-reduction active_mean \
  --opt muon --muon-lr 0.025 --adam-lr 0.008 --wd 0.01 \
  --micro-batch 256 --batch-tokens 262144 --steps 2000 --seed 42 --compile 1
