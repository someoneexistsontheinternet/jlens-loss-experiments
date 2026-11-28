#!/usr/bin/env bash
set -euo pipefail
echo "[144_verify_jlens_v3_gpu] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-verify_jlens_v3_gpu}"
PYTHONPATH=src python3 tools/verify_jlens_ablation.py
