#!/usr/bin/env bash
set -euo pipefail
echo "[050_verify_and_bench_ablation] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-verify_abl}"
PYTHONPATH=src python3 tools/verify_jlens_ablation.py
PYTHONPATH=src python3 - << 'PY'
import time, torch
from model_70m import NanoGPT70MConfig, NanoGPT70M
from jlens_ablation import CausalAblationJLensController

device = torch.device("cuda")
cfg = NanoGPT70MConfig(linear_mode="bf16", jlens_mode="ablation")
model = NanoGPT70M(cfg).to(device)
xb = torch.randint(0, 50256, (256, 1024), device=device)
yb = torch.randint(0, 50256, (256, 1024), device=device)
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    ce, nll, interm, x_raw = model(xb, yb, return_interm=True)

for b_p, k_d, win in [(32, 8, 4), (64, 8, 4), (64, 8, 1), (64, 8, 8), (128, 8, 4), (256, 16, 4)]:
    ctrl = CausalAblationJLensController(model, device=device, probe_seqs=b_p, k_dirs=k_d, window=win)
    ctrl.refresh_snapshot(model, [t.detach() for t in interm], x_raw)
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ce, nll, interm, _ = model(xb, yb, return_interm=True)
            aux, _, _ = ctrl.compute_loss(model, interm, yb, nll)
        (ce + 0.05 * aux).backward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            aux, _, st = ctrl.compute_loss(model, [t.detach().requires_grad_(True) for t in interm], yb, nll)
        aux.backward()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / 5.0
    fl_pct = 100.0 * ctrl.estimate_call_flops(256) / (262144.0 * model.count_params_and_flops()[2])
    print(f"probe_seqs={b_p:3d} k={k_d:2d} W={win} | call={ms:6.2f} ms | +FLOPs/active_step={fl_pct:5.2f}% | stats={st}", flush=True)
PY
