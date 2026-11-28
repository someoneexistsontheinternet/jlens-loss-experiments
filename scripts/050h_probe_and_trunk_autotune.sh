#!/usr/bin/env bash
set -euo pipefail
echo "[050h_probe_and_trunk_autotune] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-autotune_050h}"
PYTHONPATH=src python3 tools/verify_jlens_ablation.py
PYTHONPATH=src python3 - << 'PY'
import os, time, torch
import torch.nn.functional as F
from pathlib import Path
from model_70m import NanoGPT70MConfig, NanoGPT70M
from jlens_ablation import CausalAblationJLensController, _softcap_selected_base_bwd_kernel
from fast_kernels import extract_clean_kv_fast, advance_local_window_kv_fast, fast_clip_grad_norm_
from ultra_probe_kernels import advance_local_window_kv_ultra, _softcap_selected_bwd_2half_kernel
from train import HybridMuonAdamW
from telemetry import CompressedJSONLLogger

device = torch.device("cuda")
torch.set_float32_matmul_precision("high")
torch._dynamo.config.cache_size_limit = 128
PEAK_BF16_TFLOPS = 1307.4

run_id = os.environ.get("RUN_ID", "050h_probe_and_trunk_autotune")
sha7   = os.environ.get("GIT_COMMIT_SHA", "local")[:7]
log_p  = Path(os.environ.get("DATA_ROOT", "/data")) / "logs" / f"{run_id}_{sha7}.jsonl.gz"
logger = CompressedJSONLLogger(log_p, {"kind": "ablation_bench", "linear_mode": "bf16", "jlens_mode": "ablation"})

def bench_ms(fn, warmup=3, reps=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / reps

xb = torch.randint(0, 50256, (256, 1024), device=device)
yb = torch.randint(0, 50256, (256, 1024), device=device)

print("=" * 106, flush=True)
print(" 1. TRUNK CE_CHUNK_SZ & COMPILE-MODE SHOOTOUT (with TF32='high' + FusedBlockOps=1)", flush=True)
print("=" * 106, flush=True)

best_trunk_ms, best_csz = 1e9, 16384
for csz in (8192, 16384, 32768):
    cfg_t = NanoGPT70MConfig(linear_mode="bf16", jlens_mode="ablation", ce_chunk_sz=csz, fused_block_ops=True)
    m_t = NanoGPT70M(cfg_t).to(device)
    _, _, flops_tok = m_t.count_params_and_flops()
    step_tf = (262144.0 * flops_tok) / 1e12
    fm_t = torch.compile(m_t)
    opt_t = HybridMuonAdamW(m_t, "muon", 0.025, 0.008, 0.01)
    def st_fn():
        opt_t.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ce, _ = fm_t(xb, yb, return_interm=False)
        ce.backward()
        fast_clip_grad_norm_(m_t.parameters(), 1.0)
        opt_t.step(1.0)
    ms = bench_ms(st_fn, warmup=4, reps=8)
    tps = 262144.0 / (ms / 1000.0)
    tfl = step_tf / (ms / 1000.0)
    print(f"  [Trunk ce_csz={csz:5d}] {ms:6.2f} ms/st | {tps/1e3:6.1f}k tok/s | {tfl:5.1f} TF ({100*tfl/PEAK_BF16_TFLOPS:.2f}% MFU)", flush=True)
    if ms < best_trunk_ms:
        best_trunk_ms, best_csz = ms, csz
    del fm_t, opt_t, m_t
    torch.cuda.empty_cache()

print("=" * 106, flush=True)
print(" 2. AUTOTUNING FUSED PREFIX+LOCAL WINDOW (`advance_local_window_kv_ultra`) & PARITY CHECK", flush=True)
print("=" * 106, flush=True)

cfg = NanoGPT70MConfig(linear_mode="bf16", jlens_mode="ablation", ce_chunk_sz=best_csz, fused_block_ops=True)
raw_model = NanoGPT70M(cfg).to(device)
_, _, flops_tok = raw_model.count_params_and_flops()
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    _, nll_b, interm_b, xraw_b = raw_model(xb, yb, return_interm=True)

ctrl_t = CausalAblationJLensController(raw_model, device=device, active_frac=0.06)
b_each, pos_each, uniq_b, kv_slot = ctrl_t.select_probe_positions(yb, nll_b)
pos_each[0] = 0  # include pos=0 edge case!
N_t = 8192
p_t, s_t = pos_each[:N_t], kv_slot[:N_t]
kv_all = extract_clean_kv_fast(raw_model.blocks[3:], [t.detach() for t in interm_b[2:]], uniq_b, raw_model.cos, raw_model.sin)

h_w_anc = torch.randn((N_t, 10, 4, 512), device=device, dtype=torch.bfloat16)
h_w_ear = torch.randn((N_t, 10, 1, 512), device=device, dtype=torch.bfloat16)

ref_anc = advance_local_window_kv_fast(raw_model.blocks[9:], kv_all[6:], h_w_anc, p_t, raw_model.cos, raw_model.sin, kv_slot=s_t)
ref_ear = advance_local_window_kv_fast(raw_model.blocks[3:9], kv_all[:6], h_w_ear, p_t, raw_model.cos, raw_model.sin, kv_slot=s_t)
ms_anc_050g = bench_ms(lambda: advance_local_window_kv_fast(raw_model.blocks[9:], kv_all[6:], h_w_anc, p_t, raw_model.cos, raw_model.sin, kv_slot=s_t), warmup=3, reps=6)
ms_ear_050g = bench_ms(lambda: advance_local_window_kv_fast(raw_model.blocks[3:9], kv_all[:6], h_w_ear, p_t, raw_model.cos, raw_model.sin, kv_slot=s_t), warmup=3, reps=6)
print(f"  [050g Baseline @ N=8192] Anchor(3 blk, W=4)={ms_anc_050g:.2f}ms | Early(6 blk, W=1)={ms_ear_050g:.2f}ms | Sum={ms_anc_050g+ms_ear_050g:.2f}ms", flush=True)

best_anc_cfg, best_anc_ms = None, 1e9
best_ear_cfg, best_ear_ms = None, 1e9

for bt in (64, 128):
    for nw in (4, 8):
        for wpe in (0, 2, 4):
            uo_a = advance_local_window_kv_ultra(raw_model.blocks[9:], kv_all[6:], h_w_anc, p_t, raw_model.cos, raw_model.sin, s_t, block_t=bt, num_warps=nw, waves_per_eu=wpe)
            uo_e = advance_local_window_kv_ultra(raw_model.blocks[3:9], kv_all[:6], h_w_ear, p_t, raw_model.cos, raw_model.sin, s_t, block_t=bt, num_warps=nw, waves_per_eu=wpe)
            ca = F.cosine_similarity(ref_anc.float().reshape(1, -1), uo_a.float().reshape(1, -1)).item()
            ce_s = F.cosine_similarity(ref_ear.float().reshape(1, -1), uo_e.float().reshape(1, -1)).item()
            ma = bench_ms(lambda b=bt, w=nw, e=wpe: advance_local_window_kv_ultra(raw_model.blocks[9:], kv_all[6:], h_w_anc, p_t, raw_model.cos, raw_model.sin, s_t, block_t=b, num_warps=w, waves_per_eu=e), warmup=2, reps=6)
            me = bench_ms(lambda b=bt, w=nw, e=wpe: advance_local_window_kv_ultra(raw_model.blocks[3:9], kv_all[:6], h_w_ear, p_t, raw_model.cos, raw_model.sin, s_t, block_t=b, num_warps=w, waves_per_eu=e), warmup=2, reps=6)
            print(f"  [Ultra BT={bt:3d} NW={nw} WPE={wpe}] cos=({ca:.6f},{ce_s:.6f}) | Anc={ma:5.2f}ms ({ms_anc_050g/ma:.2f}x) | Ear={me:5.2f}ms ({ms_ear_050g/me:.2f}x) | Sum={ma+me:5.2f}ms ({(ms_anc_050g+ms_ear_050g)/(ma+me):.2f}x)", flush=True)
            assert ca > 0.9999 and ce_s > 0.9999, f"Parity failed: ca={ca}, ce={ce_s}"
            if ma < best_anc_ms:
                best_anc_ms, best_anc_cfg = ma, (bt, nw, wpe)
            if me < best_ear_ms:
                best_ear_ms, best_ear_cfg = me, (bt, nw, wpe)

print(f"  >>> Best Anchor cfg={best_anc_cfg} ({best_anc_ms:.2f}ms) | Best Early cfg={best_ear_cfg} ({best_ear_ms:.2f}ms)", flush=True)

# Test 2-half BF16-register `_softcap_selected_bwd_2half_kernel` vs loop `_softcap_selected_base_bwd_kernel`:
buf0 = torch.randn((16384, 50304), device=device, dtype=torch.bfloat16)
buf1 = buf0.clone()
lse_t = torch.full((16384,), 11.0, device=device, dtype=torch.float32)
sg_t  = torch.randn((16384,), device=device, dtype=torch.float32)
_softcap_selected_base_bwd_kernel[(16384,)](buf0, buf0.stride(0), lse_t, sg_t, V=50304, SOFTCAP=30.0, INV_SOFTCAP=1.0/30.0, BLOCK_V=4096, num_warps=8)
_softcap_selected_bwd_2half_kernel[(16384,)](buf1, buf1.stride(0), lse_t, sg_t, V=50304, SOFTCAP=30.0, INV_SOFTCAP=1.0/30.0, HALF_V=32768, num_warps=8)
bwd_cos = F.cosine_similarity(buf0.float().reshape(1, -1), buf1.float().reshape(1, -1)).item()
ms_sb_old = bench_ms(lambda: _softcap_selected_base_bwd_kernel[(16384,)](buf0, buf0.stride(0), lse_t, sg_t, V=50304, SOFTCAP=30.0, INV_SOFTCAP=1.0/30.0, BLOCK_V=4096, num_warps=8))
ms_sb_new = bench_ms(lambda: _softcap_selected_bwd_2half_kernel[(16384,)](buf1, buf1.stride(0), lse_t, sg_t, V=50304, SOFTCAP=30.0, INV_SOFTCAP=1.0/30.0, HALF_V=32768, num_warps=8))
print(f"  [SelectedBwd 16384x50304] cos={bwd_cos:.6f} | loop4096={ms_sb_old:.2f}ms -> 2half_reg={ms_sb_new:.2f}ms ({ms_sb_old/ms_sb_new:.2f}x)", flush=True)
assert bwd_cos > 0.99999

# Benchmark Oblique Suppression eager vs compiled:
x_ob = torch.randn((8192, 512), device=device, dtype=torch.bfloat16)
w_ob = torch.randn((8192, 8, 512), device=device, dtype=torch.float32)
d_ob = torch.randn((8192, 8, 512), device=device, dtype=torch.float32)
v_ob = torch.ones((8192, 8), device=device, dtype=torch.bool)
import jlens_ablation
ms_ob_eag = bench_ms(lambda: jlens_ablation.suppress_oblique_single_and_joint(x_ob, w_ob, d_ob, v_ob), warmup=2, reps=6)
comp_ob = torch.compile(jlens_ablation.suppress_oblique_single_and_joint, dynamic=True)
pr_e, uk_e, ua_e = jlens_ablation.suppress_oblique_single_and_joint(x_ob, w_ob, d_ob, v_ob)
pr_c, uk_c, ua_c = comp_ob(x_ob, w_ob, d_ob, v_ob)
ob_cos = F.cosine_similarity(pr_e.float().reshape(1, -1), pr_c.float().reshape(1, -1)).item()
ms_ob_cmp = bench_ms(lambda: comp_ob(x_ob, w_ob, d_ob, v_ob), warmup=3, reps=6)
print(f"  [ObliqueSuppress N=8192] cos={ob_cos:.6f} | eager={ms_ob_eag:.2f}ms -> compiled={ms_ob_cmp:.2f}ms ({ms_ob_eag/ms_ob_cmp:.2f}x)", flush=True)

# --- Section 3: End-to-End Full Step + Probe MFU with Ultra Kernels Hooked In! ---
print("=" * 106, flush=True)
print(" 3. END-TO-END STEP + ULTRA PROBE MFU ACROSS 0.5%, 2%, 6%, 15% ACTIVE TOKENS", flush=True)
print("=" * 106, flush=True)
def patched_adv(blocks, kv_list, h_win, positions, cos_full, sin_full, kv_slot):
    bt, nw, wpe = best_anc_cfg if h_win.shape[2] > 1 else best_ear_cfg
    return advance_local_window_kv_ultra(blocks, kv_list, h_win, positions, cos_full, sin_full, kv_slot, block_t=bt, num_warps=nw, waves_per_eu=wpe)
jlens_ablation.advance_local_window_kv_fast = patched_adv
if ms_ob_cmp < ms_ob_eag and ob_cos > 0.9999:
    jlens_ablation.suppress_oblique_single_and_joint = comp_ob

fwd_model = torch.compile(raw_model)
opt = HybridMuonAdamW(raw_model, "muon", 0.025, 0.008, 0.01)
for _ in range(3):
    opt.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        ce, _ = fwd_model(xb, yb, return_interm=False)
    ce.backward()
    fast_clip_grad_norm_(raw_model.parameters(), 1.0)
    opt.step(1.0)

for label, a_fr, ms_050d, ms_050g in [
    ("act005_ev2", 0.005, 431.9, 410.1),
    ("act020_ev2", 0.020, 472.8, 442.3),
    ("act060_ev2", 0.060, 584.0, 525.7),
    ("act150_ev2", 0.150, 839.1, 717.1),
]:
    ctrl = CausalAblationJLensController(raw_model, device=device, active_frac=a_fr, k_dirs=8, window=4)
    ctrl.refresh_snapshot(raw_model, [t.detach() for t in interm_b], xraw_b)
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
    run_avg_ms = 0.45 * best_trunk_ms + 0.55 * eff_ms
    probe_only_ms = max(1e-3, 2.0 * (eff_ms - best_trunk_ms))
    call_tf = ctrl.estimate_call_flops(256, st_last["n_act"], st_last["n_seqs"]) / 1e12
    probe_tf = call_tf / (probe_only_ms / 1000.0)
    probe_mfu = 100.0 * probe_tf / PEAK_BF16_TFLOPS
    wall_tax = 100.0 * (run_avg_ms - best_trunk_ms) / best_trunk_ms
    print(f"  [{label}] run_avg={run_avg_ms:6.1f}ms ({ms_050d/run_avg_ms:.2f}x vs 050d, +{wall_tax:5.2f}% tax) | probe={probe_only_ms:6.1f}ms ({probe_tf:5.1f} TF = {probe_mfu:5.1f}% MFU)", flush=True)
    logger.write({"type": "bench_ultra", "label": label, "run_avg_ms": round(run_avg_ms, 2), "probe_only_ms": round(probe_only_ms, 2), "probe_tflops": round(probe_tf, 1), "probe_mfu_pct": round(probe_mfu, 2)})

logger.close({"kind": "ablation_bench", "linear_mode": "bf16", "jlens_mode": "ablation", "ms_per_step": round(best_trunk_ms, 2)})
print(f"Saved -> {log_p}", flush=True)
PY
