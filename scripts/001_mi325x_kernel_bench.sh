#!/usr/bin/env bash
set -euo pipefail
echo "[001_mi325x_kernel_bench] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-bench}"
python3 -c "import torch, triton; print('PyTorch:', torch.__version__, 'HIP:', getattr(torch.version, 'hip', None), 'Triton:', triton.__version__, 'GPU:', torch.cuda.get_device_name(0))"
PYTHONPATH=src python3 src/bench_kernels.py --log-dir "${DATA_ROOT:-/data}/logs"
