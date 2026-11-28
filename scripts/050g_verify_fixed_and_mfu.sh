#!/usr/bin/env bash
set -euo pipefail
echo "[050g_verify_fixed_and_mfu] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-verify_fixed_mfu}"
PYTHONPATH=src python3 tools/verify_jlens_ablation.py
PYTHONPATH=src python3 - << 'PY'
import os, time, torch
from pathlib import Path
from model_70m import NanoGPT70MConfig, NanoGPT70M
from jlens_ablation import CausalAblationJLensController
from train import HybridMuonAdamW
from fast_kernels import fast_clip_grad_norm_
from telemetry import CompressedJSONLLogger

device = torch.device("cuda")
torch._dynamo.config.cache_size_limit = 128
PEAK_BF16_TFLOPS = 1307.4

run_id = os.environ.get("RUN_ID", "050g_verify_fixed_and_mfu")
sha7   = os.environ.get("GIT_COMMIT_SHA", "local")[:7]
log_p  = Path(os.environ.get("DATA_ROOT", "/data")) / "logs" / f"{run_id}_{sha7}.jsonl.gz"
logger = CompressedJSONLLogger(log_p, {"kind": "ablation_bench", "linear_mode": "bf16", "jlens_mode": "ablation"})

xb = torch.randint(0, 50256, (256, 1024), device=device)
yb = torch.randint(0, 50256, (256, 1024), device=device)

print("=" * 104, flush=True)
print(" 1. 70M BF16 TRUNK + COMPILED MUON MFU SHOOTOUT (Comparing FusedBlockOps=1 vs 0 & CE Chunk Size)", flush=True)
print("=" * 104, flush=True)

best_ms, best_cfg_kw = 1e9, None
for fused_ops in (True, False):
    for csz in (4096, 8192):
        cfg_t = NanoGPT70MConfig(linear_mode="bf16", jlens_mode="ablation", ce_chunk_sz=csz, fused_block_ops=fused_ops)
        m_t = NanoGPT70M(cfg_t).to(device)
        _, _, flops_tok = m_t.count_params_and_flops()
        step_tf = (262144.0 * flops_tok) / 1e12
        fm_t = torch.compile(m_t)
        opt_t = HybridMuonAdamW(m_t, "muon", 0.025, 0.008, 0.01)
        for _ in range(4):
            opt_t.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                ce, _ = fm_t(xb, yb, return_interm=False)
            ce.backward()
            fast_clip_grad_norm_(m_t.parameters(), 1.0)
            opt_t.step(1.0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(8):
            opt_t.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                ce, _ = fm_t(xb, yb, return_interm=False)
            ce.backward()
            fast_clip_grad_norm_(m_t.parameters(), 1.0)
            opt_t.step(1.0)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0 / 8.0
        tps = 262144.0 / (ms / 1000.0)
        tfl = step_tf / (ms / 1000.0)
        mfu = 100.0 * tfl / PEAK_BF16_TFLOPS
        print(f"  [Trunk fused_ops={int(fused_ops)} ce_csz={csz:4d}] {ms:6.2f} ms/st (vs 402.0ms old = {402.0/ms:.2f}x) | {tps/1e3:6.1f}k tok/s | {tfl:5.1f} TF ({mfu:.1f}% MFU)", flush=True)
        logger.write({"type": "bench_trunk", "fused_ops": int(fused_ops), "ce_csz": csz, "ms_per_step": round(ms, 2), "tok_per_s": round(tps, 1), "tflops": round(tfl, 1), "mfu_pct": round(mfu, 2)})
        if ms < best_ms:
            best_ms = ms
            best_cfg_kw = (fused_ops, csz)
        del fm_t, opt_t, m_t
        torch.cuda.empty_cache()

print("=" * 104, flush=True)
print(f" 2. FIXED + ACCELERATED J-LENS PROBE ENGINE BENCHMARK (Using Best Trunk {best_cfg_kw} @ {best_ms:.2f}ms/st)", flush=True)
print("=" * 104, flush=True)

cfg = NanoGPT70MConfig(linear_mode="bf16", jlens_mode="ablation", ce_chunk_sz=best_cfg_kw[1], fused_block_ops=best_cfg_kw[0])
raw_model = NanoGPT70M(cfg).to(device)
_, _, flops_tok = raw_model.count_params_and_flops()
fwd_model = torch.compile(raw_model)
opt = HybridMuonAdamW(raw_model, "muon", 0.025, 0.008, 0.01)

# Warm up trunk
for _ in range(3):
    opt.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        ce, _ = fwd_model(xb, yb, return_interm=False)
    ce.backward()
    fast_clip_grad_norm_(raw_model.parameters(), 1.0)
    opt.step(1.0)

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    _, nll_init, interm_init, x_raw_init = raw_model(xb, yb, return_interm=True)

old_run_avg = {"act005_ev2": 431.9, "act020_ev2": 472.8, "act060_ev2": 584.0, "act150_ev2": 839.1}
for label, a_fr in [("act005_ev2", 0.005), ("act020_ev2", 0.020), ("act060_ev2", 0.060), ("act150_ev2", 0.150)]:
    ctrl = CausalAblationJLensController(raw_model, device=device, active_frac=a_fr, k_dirs=8, window=4)
    ctrl.refresh_snapshot(raw_model, [t.detach() for t in interm_init], x_raw_init)
    for _ in range(2):
        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ce, nll, interm, _ = fwd_model(xb, yb, return_interm=True)
            aux, _, _ = ctrl.compute_loss(raw_model, interm, yb, nll)
        (ce + 0.50 * aux).backward()
        fast_clip_grad_norm_(raw_model.parameters(), 1.0)
        opt.step(1.0)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    st_last = {}
    for s_i in range(6):
        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if s_i % 2 == 0:
                ce, nll, interm, _ = fwd_model(xb, yb, return_interm=True)
                aux, _, st_last = ctrl.compute_loss(raw_model, interm, yb, nll)
                tot = ce + 0.50 * aux
            else:
                ce, _ = fwd_model(xb, yb, return_interm=False)
                tot = ce
        tot.backward()
        fast_clip_grad_norm_(raw_model.parameters(), 1.0)
        opt.step(1.0)
    torch.cuda.synchronize()
    eff_ms = (time.perf_counter() - t0) * 1000.0 / 6.0
    run_avg_ms = 0.45 * best_ms + 0.55 * eff_ms
    probe_only_ms = max(1e-3, 2.0 * (eff_ms - best_ms))
    call_tflops = ctrl.estimate_call_flops(256, st_last["n_act"], st_last["n_seqs"]) / 1e12
    probe_tf = call_tflops / (probe_only_ms / 1000.0)
    probe_mfu = 100.0 * probe_tf / PEAK_BF16_TFLOPS
    run_fl_pct = 0.55 * (100.0 * (call_tflops * 1e12) / (262144.0 * flops_tok) / 2.0)
    wall_tax_pct = 100.0 * (run_avg_ms - best_ms) / best_ms
    sp_vs_050d = old_run_avg[label] / run_avg_ms
    print(f"  [{label}] n_act={st_last['n_act']:5d} | run_avg={run_avg_ms:6.1f}ms ({sp_vs_050d:.2f}x vs 050d, +{wall_tax_pct:5.2f}% tax) | probe={probe_only_ms:6.1f}ms ({probe_tf:5.1f} TF = {probe_mfu:4.1f}% MFU) | +FLOPs={run_fl_pct:5.2f}% | j_batch={st_last['j_batch']:.5f} | |uA|={st_last['u_anc_abs']:.4f}", flush=True)
    logger.write({
        "type": "bench_probe_fast", "label": label, "active_frac": a_fr,
        "run_avg_ms": round(run_avg_ms, 2), "probe_only_ms": round(probe_only_ms, 2),
        "probe_tflops": round(probe_tf, 1), "probe_mfu_pct": round(probe_mfu, 2),
        "wall_tax_pct": round(wall_tax_pct, 2), "extra_flops_pct": round(run_fl_pct, 3),
        **st_last,
    })

logger.close({"kind": "ablation_bench", "linear_mode": "bf16", "jlens_mode": "ablation", "ms_per_step": round(best_ms, 2)})
print(f"Saved -> {log_p}", flush=True)
PY
