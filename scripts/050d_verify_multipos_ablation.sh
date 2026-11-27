#!/usr/bin/env bash
set -euo pipefail
echo "[050d_verify_multipos_ablation] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-verify_multipos}"
PYTHONPATH=src python3 tools/verify_jlens_ablation.py
PYTHONPATH=src python3 - << 'PY'
import os, time, torch
from pathlib import Path
from model_70m import NanoGPT70MConfig, NanoGPT70M
from jlens_ablation import CausalAblationJLensController
from train import HybridMuonAdamW
from telemetry import CompressedJSONLLogger

device = torch.device("cuda")
torch._dynamo.config.cache_size_limit = 64
run_id = os.environ.get("RUN_ID", "050d_verify_multipos_ablation")
sha7   = os.environ.get("GIT_COMMIT_SHA", "local")[:7]
log_p  = Path(os.environ.get("DATA_ROOT", "/data")) / "logs" / f"{run_id}_{sha7}.jsonl.gz"
logger = CompressedJSONLLogger(log_p, {"kind": "ablation_bench", "linear_mode": "bf16", "jlens_mode": "ablation"})

cfg = NanoGPT70MConfig(linear_mode="bf16", jlens_mode="ablation")
raw_model = NanoGPT70M(cfg).to(device)
_, _, flops_tok = raw_model.count_params_and_flops()
fwd_model = torch.compile(raw_model)
opt = HybridMuonAdamW(raw_model, "muon", 0.025, 0.008, 0.01)
xb = torch.randint(0, 50256, (256, 1024), device=device)
yb = torch.randint(0, 50256, (256, 1024), device=device)

for _ in range(3):
    opt.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        ce, _ = fwd_model(xb, yb, return_interm=False)
    ce.backward()
    opt.step(1.0)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(6):
    opt.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        ce, _ = fwd_model(xb, yb, return_interm=False)
    ce.backward()
    opt.step(1.0)
torch.cuda.synchronize()
base_ms = (time.perf_counter() - t0) * 1000.0 / 6.0
print(f"[Baseline BF16 Step] {base_ms:.2f} ms/st ({262144.0/(base_ms/1000.0)/1e3:.1f}k tok/s)", flush=True)
logger.write({"type": "bench_base", "ms_per_step": round(base_ms, 2)})

with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    _, nll_init, interm_init, x_raw_init = raw_model(xb, yb, return_interm=True)

configs = [
    ("act005_ev2_k8_W4", 0.005, 8, 4, 2),
    ("act020_ev2_k8_W4", 0.020, 8, 4, 2),
    ("act060_ev2_k8_W4", 0.060, 8, 4, 2),
    ("act150_ev2_k8_W4", 0.150, 8, 4, 2),
    ("act020_ev1_k8_W4", 0.020, 8, 4, 1),
    ("act060_ev1_k8_W4", 0.060, 8, 4, 1),
]

for label, a_fr, k_d, win, ev in configs:
    ctrl = CausalAblationJLensController(raw_model, device=device, active_frac=a_fr, k_dirs=k_d, window=win)
    ctrl.refresh_snapshot(raw_model, [t.detach() for t in interm_init], x_raw_init)
    for _ in range(2):
        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ce, nll, interm, _ = fwd_model(xb, yb, return_interm=True)
            aux, _, _ = ctrl.compute_loss(raw_model, interm, yb, nll)
        (ce + 0.05 * aux).backward()
        opt.step(1.0)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    st_last = {}
    for s_i in range(6):
        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if s_i % ev == 0:
                ce, nll, interm, _ = fwd_model(xb, yb, return_interm=True)
                aux, _, st_last = ctrl.compute_loss(raw_model, interm, yb, nll)
                tot = ce + 0.05 * aux
            else:
                ce, _ = fwd_model(xb, yb, return_interm=False)
                tot = ce
        tot.backward()
        opt.step(1.0)
    torch.cuda.synchronize()
    eff_ms = (time.perf_counter() - t0) * 1000.0 / 6.0
    run_avg_ms = 0.45 * base_ms + 0.55 * eff_ms
    wall_tax_pct = 100.0 * (run_avg_ms - base_ms) / base_ms
    active_fl_pct = 100.0 * ctrl.estimate_call_flops(256, st_last["n_act"], st_last["n_seqs"]) / (262144.0 * flops_tok)
    run_fl_pct = 0.55 * (active_fl_pct / ev)
    mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    rec = {
        "type": "bench_multipos", "label": label, "active_frac": a_fr, "k": k_d, "W": win, "every": ev,
        "active_cadence_ms": round(eff_ms, 2), "run_avg_ms": round(run_avg_ms, 2),
        "wall_tax_pct": round(wall_tax_pct, 2), "extra_flops_pct": round(run_fl_pct, 3),
        "peak_mem_gb": round(mem_gb, 1), **st_last,
    }
    logger.write(rec)
    print(f"  {label:18s} | n_act={st_last['n_act']:5d} | run_avg={run_avg_ms:6.1f} ms (+{wall_tax_pct:5.2f}% time) | +FLOPs={run_fl_pct:6.2f}% | mem={mem_gb:4.1f}GB | |uA|={st_last['u_anc_abs']:.4f}", flush=True)

logger.close({"kind": "ablation_bench", "linear_mode": "bf16", "jlens_mode": "ablation", "ms_per_step": round(base_ms, 2)})
print(f"Saved -> {log_p}", flush=True)
PY
