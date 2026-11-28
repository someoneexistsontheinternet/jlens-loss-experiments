#!/usr/bin/env bash
set -euo pipefail
echo "[120b_verify_jlens_v2_gpu] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-verify_v2_gpu}"
PYTHONPATH=src python3 tools/verify_jlens_ablation.py
