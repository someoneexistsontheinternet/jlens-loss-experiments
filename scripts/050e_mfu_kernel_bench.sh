#!/usr/bin/env bash
set -euo pipefail
echo "[050e_mfu_kernel_bench] Commit: ${GIT_COMMIT_SHA:-local} | Run: ${RUN_ID:-mfu_bench}"
PYTHONPATH=src python3 - << 'PY'
import math, os, time, torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from model_70m import NanoGPT70MConfig, NanoGPT70M, FastRMSNorm, apply_rope, FusedTritonChunkedSoftcapCEFn
from fast_kernels import (
    FusedRMSNormFn, FusedQKNormRoPEFn, FusedSwiGLUFn, FastChunkedSoftcapCEFn,
    compiled_newton_schulz_5, fast_clip_grad_norm_,
    extract_clean_kv_fast, _grouped_flash_prefix_kv_kernel, build_seq_groups,
    advance_local_window_kv_fast,
)
from jlens_ablation import (
    extract_clean_kv_for_blocks, advance_local_window_kv, _flash_prefix_kv_kernel,
    CausalAblationJLensController,
)
from train import HybridMuonAdamW, newton_schulz_5
from telemetry import CompressedJSONLLogger
import triton

device = torch.device("cuda")
torch._dynamo.config.cache_size_limit = 128
torch.manual_seed(42)
run_id = os.environ.get("RUN_ID", "050e_mfu_kernel_bench")
sha7   = os.environ.get("GIT_COMMIT_SHA", "local")[:7]
log_p  = Path(os.environ.get("DATA_ROOT", "/data")) / "logs" / f"{run_id}_{sha7}.jsonl.gz"
logger = CompressedJSONLLogger(log_p, {"kind": "ablation_bench", "linear_mode": "bf16", "jlens_mode": "ablation"})

PEAK_BF16_TFLOPS = 1307.4  # AMD Instinct MI325X dense BF16 peak

def bench_ms(fn, warmup=4, reps=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / reps

print("=" * 100)
print(" 1. NUMERICAL PARITY & MICRO-BENCHMARK OF FUSED TRITON KERNELS ON MI325X (gfx942)")
print("=" * 100)

# --- A1: FusedRMSNorm ---
B, T, D, H, Dh, F_d, V = 256, 1024, 512, 8, 64, 1792, 50304
x_ref = torch.randn((B, T, D), device=device, dtype=torch.bfloat16, requires_grad=True)
x_opt = x_ref.detach().clone().requires_grad_(True)
rn_ref = FastRMSNorm(D).to(device)
w_opt = rn_ref.weight.detach().clone().requires_grad_(True)
dy_rn = torch.randn_like(x_ref)

y_r = rn_ref(x_ref); y_r.backward(dy_rn)
y_o = FusedRMSNormFn.apply(x_opt, w_opt, 1e-6); y_o.backward(dy_rn)
rn_fwd_err = (y_r.float() - y_o.float()).abs().max().item()
rn_dx_cos  = F.cosine_similarity(x_ref.grad.float().reshape(1, -1), x_opt.grad.float().reshape(1, -1)).item()
rn_dw_cos  = F.cosine_similarity(rn_ref.weight.grad.float().reshape(1, -1), w_opt.grad.float().reshape(1, -1)).item()
assert rn_fwd_err <= 2e-2 and rn_dx_cos > 0.9999 and rn_dw_cos > 0.99999
ms_rn_eager = bench_ms(lambda: (rn_ref(x_ref).backward(dy_rn), x_ref.grad.zero_(), rn_ref.weight.grad.zero_()))
ms_rn_trit  = bench_ms(lambda: (FusedRMSNormFn.apply(x_opt, w_opt, 1e-6).backward(dy_rn), x_opt.grad.zero_(), w_opt.grad.zero_()))
print(f"[A1 FusedRMSNorm]      fwd_max={rn_fwd_err:.5f} dx_cos={rn_dx_cos:.6f} dw_cos={rn_dw_cos:.6f} | eager={ms_rn_eager:.2f}ms -> triton={ms_rn_trit:.2f}ms ({ms_rn_eager/ms_rn_trit:.2f}x)")

# --- A2: FusedQKNormRoPE ---
qkv_ref = torch.randn((B, T, 3, H, Dh), device=device, dtype=torch.bfloat16, requires_grad=True)
qkv_opt = qkv_ref.detach().clone().requires_grad_(True)
qn_mod, kn_mod = FastRMSNorm(Dh).to(device), FastRMSNorm(Dh).to(device)
wq_o = qn_mod.weight.detach().clone().requires_grad_(True)
wk_o = kn_mod.weight.detach().clone().requires_grad_(True)
freqs = torch.outer(torch.arange(T, device=device, dtype=torch.float32), 1.0 / (10000.0 ** (torch.arange(0, 32, device=device, dtype=torch.float32) / 32.0)))
cos_b = torch.cos(freqs)[None, None, :, :].to(torch.bfloat16)
sin_b = torch.sin(freqs)[None, None, :, :].to(torch.bfloat16)
dq_t = torch.randn((B, H, T, Dh), device=device, dtype=torch.bfloat16)
dk_t = torch.randn((B, H, T, Dh), device=device, dtype=torch.bfloat16)
dv_t = torch.randn((B, H, T, Dh), device=device, dtype=torch.bfloat16)

def ref_qknorm_rope(qkv):
    q = apply_rope(qn_mod(qkv[:, :, 0]).transpose(1, 2), cos_b, sin_b)
    k = apply_rope(kn_mod(qkv[:, :, 1]).transpose(1, 2), cos_b, sin_b)
    v = qkv[:, :, 2].transpose(1, 2)
    return q, k, v

qr, kr, vr = ref_qknorm_rope(qkv_ref)
torch.autograd.backward((qr, kr, vr), (dq_t, dk_t, dv_t))
qo, ko, vo = FusedQKNormRoPEFn.apply(qkv_opt, wq_o, wk_o, cos_b, sin_b, 1e-6)
torch.autograd.backward((qo, ko, vo), (dq_t, dk_t, dv_t))
qk_fwd_err = max((qr.float() - qo.float()).abs().max().item(), (kr.float() - ko.float()).abs().max().item(), (vr.float() - vo.float()).abs().max().item())
qk_dx_cos  = F.cosine_similarity(qkv_ref.grad.float().reshape(1, -1), qkv_opt.grad.float().reshape(1, -1)).item()
qk_dw_cos  = F.cosine_similarity(qn_mod.weight.grad.float().reshape(1, -1), wq_o.grad.float().reshape(1, -1)).item()
assert qk_fwd_err <= 2e-2 and qk_dx_cos > 0.9999 and qk_dw_cos > 0.99999
comp_ref_qk = torch.compile(ref_qknorm_rope)
ms_qk_comp = bench_ms(lambda: (torch.autograd.backward(comp_ref_qk(qkv_ref), (dq_t, dk_t, dv_t)), qkv_ref.grad.zero_()))
ms_qk_trit = bench_ms(lambda: (torch.autograd.backward(FusedQKNormRoPEFn.apply(qkv_opt, wq_o, wk_o, cos_b, sin_b, 1e-6), (dq_t, dk_t, dv_t)), qkv_opt.grad.zero_()))
print(f"[A2 FusedQKNormRoPE]   fwd_max={qk_fwd_err:.5f} dx_cos={qk_dx_cos:.6f} dw_cos={qk_dw_cos:.6f} | compiled={ms_qk_comp:.2f}ms -> triton={ms_qk_trit:.2f}ms ({ms_qk_comp/ms_qk_trit:.2f}x)")

# Check SDPA speed with non-contiguous v vs contiguous v!
ms_sdpa_strided = bench_ms(lambda: torch.autograd.backward(F.scaled_dot_product_attention(qr.detach().requires_grad_(True), kr.detach().requires_grad_(True), vr.detach().requires_grad_(True), is_causal=True), dq_t))
ms_sdpa_contig  = bench_ms(lambda: torch.autograd.backward(F.scaled_dot_product_attention(qo.detach().requires_grad_(True), ko.detach().requires_grad_(True), vo.detach().requires_grad_(True), is_causal=True), dq_t))
print(f"[SDPA Strided vs Contig V] strided_v={ms_sdpa_strided:.2f}ms vs contig_v={ms_sdpa_contig:.2f}ms ({ms_sdpa_strided/ms_sdpa_contig:.2f}x)")

# --- A3: FusedSwiGLU ---
gu_ref = torch.randn((B * T, 2 * F_d), device=device, dtype=torch.bfloat16, requires_grad=True)
gu_opt = gu_ref.detach().clone().requires_grad_(True)
dy_sw  = torch.randn((B * T, F_d), device=device, dtype=torch.bfloat16)
def ref_swiglu(gu):
    g, u = gu.chunk(2, dim=-1)
    return F.silu(g) * u
sr = ref_swiglu(gu_ref); sr.backward(dy_sw)
so = FusedSwiGLUFn.apply(gu_opt); so.backward(dy_sw)
sw_err = (sr.float() - so.float()).abs().max().item()
sw_cos = F.cosine_similarity(gu_ref.grad.float().reshape(1, -1), gu_opt.grad.float().reshape(1, -1)).item()
assert sw_err <= 5e-2 and sw_cos > 0.9999
comp_sw = torch.compile(ref_swiglu)
ms_sw_comp = bench_ms(lambda: (comp_sw(gu_ref).backward(dy_sw), gu_ref.grad.zero_()))
ms_sw_trit = bench_ms(lambda: (FusedSwiGLUFn.apply(gu_opt).backward(dy_sw), gu_opt.grad.zero_()))
print(f"[A3 FusedSwiGLU]       fwd_max={sw_err:.5f} dx_cos={sw_cos:.6f} | compiled={ms_sw_comp:.2f}ms -> triton={ms_sw_trit:.2f}ms ({ms_sw_comp/ms_sw_trit:.2f}x)")

# --- A4: Softcap-CE Chunk Size & 1-Pass VGPR Kernel ---
h_ce = torch.randn((B * T, D), device=device, dtype=torch.bfloat16, requires_grad=True)
w_ce = torch.randn((V, D), device=device, dtype=torch.float32, requires_grad=True) * 0.02
t_ce = torch.randint(0, 50256, (B * T,), device=device)
l_old, _ = FusedTritonChunkedSoftcapCEFn.apply(h_ce, w_ce, t_ce, 30.0, 32768); l_old.backward()
dh_old, dw_old = h_ce.grad.clone(), w_ce.grad.clone()
h_ce.grad = None; w_ce.grad = None
l_new, _ = FastChunkedSoftcapCEFn.apply(h_ce, w_ce, t_ce, 30.0, 4096); l_new.backward()
ce_diff = abs(l_old.item() - l_new.item())
ce_dh_cos = F.cosine_similarity(dh_old.float().reshape(1, -1), h_ce.grad.float().reshape(1, -1)).item()
ce_dw_cos = F.cosine_similarity(dw_old.float().reshape(1, -1), w_ce.grad.float().reshape(1, -1)).item()
assert ce_diff < 1e-4 and ce_dh_cos > 0.99999 and ce_dw_cos > 0.9999
ms_ce_old = bench_ms(lambda: (FusedTritonChunkedSoftcapCEFn.apply(h_ce, w_ce, t_ce, 30.0, 32768)[0].backward(), setattr(h_ce, 'grad', None), setattr(w_ce, 'grad', None)))
for use_1p in (True, False):
    for csz in (2048, 4096, 8192):
        ms_c = bench_ms(lambda c=csz, u1=use_1p: (FastChunkedSoftcapCEFn.apply(h_ce, w_ce, t_ce, 30.0, c, u1)[0].backward(), setattr(h_ce, 'grad', None), setattr(w_ce, 'grad', None)))
        tfl = (40.51 / (ms_c / 1000.0))
        print(f"[A4 FastSoftcapCE 1p={int(use_1p)} csz={csz:5d}] diff={ce_diff:.6f} dh_cos={ce_dh_cos:.6f} | old={ms_ce_old:.2f}ms -> {ms_c:.2f}ms ({ms_ce_old/ms_c:.2f}x, {tfl:.1f} TF = {100*tfl/PEAK_BF16_TFLOPS:.1f}% MFU)")

# --- B2: Grouped Same-Sequence 128x64 MFMA Flash-Prefix Parity & Speed ---
print("=" * 100)
print(" 2. J-LENS PROBE ENGINE SUB-KERNEL PARITY & SPEEDUP")
print("=" * 100)
cfg = NanoGPT70MConfig(linear_mode="bf16", jlens_mode="ablation")
raw_model = NanoGPT70M(cfg).to(device)
xb = torch.randint(0, 50256, (256, 1024), device=device)
yb = torch.randint(0, 50256, (256, 1024), device=device)
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    _, nll_b, interm_b, xraw_b = raw_model(xb, yb, return_interm=True)

b_all = torch.arange(256, device=device)
kv_ref = extract_clean_kv_for_blocks(raw_model.blocks[3:], [t.detach() for t in interm_b[2:]], b_all, raw_model.cos, raw_model.sin)
kv_fst = extract_clean_kv_fast(raw_model.blocks[3:], [t.detach() for t in interm_b[2:]], b_all, raw_model.cos, raw_model.sin)
kv_err = max(max((kr0.float() - kf0.float()).abs().max().item(), (vr0.float() - vf0.float()).abs().max().item()) for (kr0, vr0), (kf0, vf0) in zip(kv_ref, kv_fst))
assert kv_err <= 2e-2
ms_kv_old = bench_ms(lambda: extract_clean_kv_for_blocks(raw_model.blocks[3:], [t.detach() for t in interm_b[2:]], b_all, raw_model.cos, raw_model.sin))
ms_kv_new = bench_ms(lambda: extract_clean_kv_fast(raw_model.blocks[3:], [t.detach() for t in interm_b[2:]], b_all, raw_model.cos, raw_model.sin))
print(f"[B1 extract_clean_kv 9-layers] max_err={kv_err:.5f} | old={ms_kv_old:.2f}ms -> fast={ms_kv_new:.2f}ms ({ms_kv_old/ms_kv_new:.2f}x)")

# Test Grouped Flash-Prefix on Early (PW=10, G=12) and Anchor (PW=40, G=3) at N_pos=8192 (including pos=0!)
ctrl_t = CausalAblationJLensController(raw_model, device=device, active_frac=0.06)
b_each, pos_each, uniq_b, kv_slot = ctrl_t.select_probe_positions(yb, nll_b)
pos_each[0] = 0  # explicitly test pos=0 inside group!
N_t = 8192
b_e, p_e, s_e = b_each[:N_t], pos_each[:N_t].to(torch.int32).contiguous(), kv_slot[:N_t].to(torch.int32).contiguous()
k_bf, v_bf = kv_fst[0][0], kv_fst[0][1]

for label, pw, grp_g in [("Early_PW10_G12", 10, 12), ("Anchor_PW40_G3", 40, 3)]:
    q_pw = torch.randn((N_t, H, pw, Dh), device=device, dtype=torch.bfloat16)
    o_ref = torch.empty((N_t, H, pw, Dh), dtype=torch.float32, device=device)
    l_ref = torch.empty((N_t, H, pw), dtype=torch.float32, device=device)
    o_grp = torch.empty_like(o_ref)
    l_grp = torch.empty_like(l_ref)
    bpw = max(16, triton.next_power_of_2(pw))

    def run_old_fp():
        _flash_prefix_kv_kernel[(N_t, H)](
            q_pw, k_bf, v_bf, s_e, p_e, o_ref, l_ref,
            q_pw.stride(0), q_pw.stride(1), q_pw.stride(2),
            k_bf.stride(0), k_bf.stride(1), k_bf.stride(2),
            o_ref.stride(0), o_ref.stride(1), o_ref.stride(2),
            l_ref.stride(0), l_ref.stride(1),
            PW=pw, SM_SCALE=1.0 / 8.0, BLOCK_PW=bpw, BLOCK_T=64, DH=64, num_warps=4,
        )

    def run_grp_fp():
        gs, ge = build_seq_groups(s_e, uniq_b.shape[0], grp_g)
        _grouped_flash_prefix_kv_kernel[(gs.shape[0], H)](
            q_pw, k_bf, v_bf, gs, ge, s_e, p_e, o_grp, l_grp,
            q_pw.stride(0), q_pw.stride(1), q_pw.stride(2),
            k_bf.stride(0), k_bf.stride(1), k_bf.stride(2),
            o_grp.stride(0), o_grp.stride(1), o_grp.stride(2),
            l_grp.stride(0), l_grp.stride(1),
            SM_SCALE=1.0 / 8.0, PW=pw, BLOCK_M=128, BLOCK_T=64, DH=64, num_warps=4,
        )

    run_old_fp(); run_grp_fp()
    assert torch.isfinite(o_grp).all()
    fp_err = (o_ref - o_grp).abs().max().item()
    assert fp_err < 1e-3, f"Grouped flash prefix mismatch: {fp_err}"
    ms_fp_old = bench_ms(run_old_fp)
    ms_fp_grp = bench_ms(run_grp_fp)
    print(f"[B2 {label:16s} N=8192] max_err={fp_err:.6f} | old={ms_fp_old:.2f}ms -> grouped_128x64={ms_fp_grp:.2f}ms ({ms_fp_old/ms_fp_grp:.2f}x)")

# --- End-to-End 70M Step Breakdown & Upgraded Trunk Benchmark ---
print("=" * 100)
print(" 3. END-TO-END 70M BF16 TRUNK + MUON OPTIMIZER MFU BENCHMARK")
print("=" * 100)
_, _, flops_tok = raw_model.count_params_and_flops()
step_tflops = (262144.0 * flops_tok) / 1e12

# Baseline step (exact current code)
fwd_base = torch.compile(raw_model)
opt_base = HybridMuonAdamW(raw_model, "muon", 0.025, 0.008, 0.01)
def step_current():
    opt_base.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        ce, _ = fwd_base(xb, yb, return_interm=False)
    ce.backward()
    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
    opt_base.step(1.0)

ms_cur = bench_ms(step_current, warmup=4, reps=8)
print(f"[Current 70M Step]   {ms_cur:6.2f} ms/st | {262144.0/(ms_cur/1000)/1e3:6.1f}k tok/s | {step_tflops/(ms_cur/1000):5.1f} TFLOPS ({100*(step_tflops/(ms_cur/1000))/PEAK_BF16_TFLOPS:.1f}% MFU)")

# Measure Optimizer + Clip overhead alone (current vs fast compiled):
opt_base.zero_grad()
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    ce, _ = fwd_base(xb, yb, return_interm=False)
ce.backward()
ms_opt_cur = bench_ms(lambda: (torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0), opt_base.step(1.0)), warmup=2, reps=8)

adam_fused = torch.optim.AdamW(
    [{"params": [raw_model.tok_emb.weight], "lr": 0.0064, "weight_decay": 0.01},
     {"params": [p for n, p in raw_model.named_parameters() if p.ndim < 2], "lr": 0.008, "weight_decay": 0.0}],
    betas=(0.9, 0.95), fused=True,
)
muon_by_shape = {}
for p, buf in zip(opt_base.muon_params, opt_base.muon_bufs):
    muon_by_shape.setdefault(p.shape, ([], []))[0].append(p)
    muon_by_shape[p.shape][1].append(buf)

@torch.no_grad()
def fast_opt_step():
    fast_clip_grad_norm_(raw_model.parameters(), 1.0)
    adam_fused.step()
    for _, (ps, bufs) in muon_by_shape.items():
        gs = [p.grad for p in ps]
        torch._foreach_mul_(bufs, 0.95)
        torch._foreach_add_(bufs, gs)
        nesterov = torch._foreach_add(gs, bufs, alpha=0.95)
        u_stk = compiled_newton_schulz_5(torch.stack(nesterov, dim=0))
        torch._foreach_mul_(ps, 1.0 - 0.025 * 0.01)
        torch._foreach_add_(ps, list(u_stk.unbind(0)), alpha=-0.025)

ms_opt_fast = bench_ms(fast_opt_step, warmup=4, reps=8)
print(f"[GradClip + MuonAdamW] current={ms_opt_cur:.2f}ms -> fast_compiled={ms_opt_fast:.2f}ms ({ms_opt_cur/ms_opt_fast:.2f}x)")

# Now build FastNanoGPT70M wrapper using FusedRMSNorm + FusedQKNormRoPE + FusedSwiGLU + FastChunkedSoftcapCE(csz=4096) + BF16 tok_emb:
class FastBlock(nn.Module):
    def __init__(self, blk):
        super().__init__()
        self.blk = blk
    def forward(self, x, cos, sin):
        B, T, D = x.shape
        H, Dh = self.blk.cfg.n_heads, self.blk.cfg.head_dim
        xn = FusedRMSNormFn.apply(x, self.blk.attn_norm.weight, self.blk.attn_norm.eps)
        qkv = self.blk.wqkv(xn).view(B, T, 3, H, Dh)
        q, k, v = FusedQKNormRoPEFn.apply(qkv, self.blk.q_norm.weight, self.blk.k_norm.weight, cos, sin, self.blk.q_norm.eps)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)
        x = x + self.blk.wo(y)
        mn = FusedRMSNormFn.apply(x, self.blk.mlp_norm.weight, self.blk.mlp_norm.eps)
        h  = FusedSwiGLUFn.apply(self.blk.wgu(mn))
        return x + self.blk.wdown(h)

class FastModel(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.fblocks = nn.ModuleList([FastBlock(b) for b in m.blocks])
    def forward_bb(self, idx):
        T = idx.shape[1]
        cos, sin = self.m.cos[:, :, :T, :], self.m.sin[:, :, :T, :]
        x = F.embedding(idx, self.m.tok_emb.weight.to(torch.bfloat16))
        for fb in self.fblocks:
            x = fb(x, cos, sin)
        return FusedRMSNormFn.apply(x, self.m.out_norm.weight, self.m.out_norm.eps)
    def forward(self, idx, targets):
        h = self.forward_bb(idx)
        return FastChunkedSoftcapCEFn.apply(h.reshape(-1, 512), self.m.tok_emb.weight, targets.reshape(-1), 30.0, 4096)

# Verify end-to-end 70M loss & gradient parity between raw_model and FastModel!
raw_model.zero_grad(set_to_none=True)
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    l_ref70, _ = raw_model(xb[:16], yb[:16])
l_ref70.backward()
grads_ref = {n: p.grad.detach().float().clone() for n, p in raw_model.named_parameters()}

raw_model.zero_grad(set_to_none=True)
fm = FastModel(raw_model)
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    l_fst70, _ = fm(xb[:16], yb[:16])
l_fst70.backward()
min_gcos = min(F.cosine_similarity(grads_ref[n].reshape(1, -1), p.grad.float().reshape(1, -1)).item() for n, p in raw_model.named_parameters())
print(f"[70M Full-Model Parity] loss_diff={abs(l_ref70.item()-l_fst70.item()):.6f} | min_param_grad_cos={min_gcos:.6f}")
assert abs(l_ref70.item() - l_fst70.item()) < 5e-3 and min_gcos > 0.999

# Benchmark FastModel (both eager-custom-triton and torch.compile-wrapped backbone):
fm_comp_bb = torch.compile(fm.forward_bb)
def step_fast_eager():
    opt_base.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        ce, _ = fm(xb, yb)
    ce.backward()
    fast_opt_step()

def step_fast_compiled():
    opt_base.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        h = fm_comp_bb(xb)
        ce, _ = FastChunkedSoftcapCEFn.apply(h.reshape(-1, 512), raw_model.tok_emb.weight, yb.reshape(-1), 30.0, 4096)
    ce.backward()
    fast_opt_step()

# Also test current NanoGPTBlock with ONLY contiguous V + FastChunkedSoftcapCE(4096) + fast_opt_step under torch.compile!
def bb_contig_v(idx):
    T = idx.shape[1]
    cos, sin = raw_model.cos[:, :, :T, :], raw_model.sin[:, :, :T, :]
    x = raw_model.tok_emb(idx).to(torch.bfloat16)
    for blk in raw_model.blocks:
        B_, T_, D_ = x.shape
        qkv = blk.wqkv(blk.attn_norm(x)).view(B_, T_, 3, 8, 64)
        q = apply_rope(blk.q_norm(qkv[:, :, 0]).transpose(1, 2), cos, sin)
        k = apply_rope(blk.k_norm(qkv[:, :, 1]).transpose(1, 2), cos, sin)
        v = qkv[:, :, 2].transpose(1, 2).contiguous()
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B_, T_, D_)
        x = x + blk.wo(y)
        g, u = blk.wgu(blk.mlp_norm(x)).chunk(2, dim=-1)
        x = x + blk.wdown(F.silu(g) * u)
    return raw_model.out_norm(x)

comp_contig_v = torch.compile(bb_contig_v)
def step_comp_contig_v():
    opt_base.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        h = comp_contig_v(xb)
        ce, _ = FastChunkedSoftcapCEFn.apply(h.reshape(-1, 512), raw_model.tok_emb.weight, yb.reshape(-1), 30.0, 4096)
    ce.backward()
    fast_opt_step()

for name, fn in [
    ("FastTriton_EagerBB", step_fast_eager),
    ("FastTriton_CompiledBB", step_fast_compiled),
    ("InductorContigV+FastCE+FastOpt", step_comp_contig_v),
]:
    ms = bench_ms(fn, warmup=4, reps=8)
    tps = 262144.0 / (ms / 1000.0)
    tfl = step_tflops / (ms / 1000.0)
    print(f"[{name:30s}] {ms:6.2f} ms/st ({ms_cur/ms:.2f}x) | {tps/1e3:6.1f}k tok/s | {tfl:5.1f} TFLOPS ({100*tfl/PEAK_BF16_TFLOPS:.1f}% MFU)")

# --- Section 4: Full `advance_local_window_kv_fast` & Probe-Engine Sub-Component Breakdown ---
print("=" * 100)
print(" 4. FULL PROBE ENGINE BREAKDOWN & `advance_local_window_kv_fast` SPEEDUP (N_act=5243 & 15729)")
print("=" * 100)
for n_test, tag in [(5243, "act020_5243"), (8192, "chunk_8192")]:
    b_t = b_each[:n_test]
    p_t = pos_each[:n_test]
    s_t = kv_slot[:n_test]
    h_w_anc = torch.randn((n_test, 10, 4, 512), device=device, dtype=torch.bfloat16)
    h_w_ear = torch.randn((n_test, 10, 1, 512), device=device, dtype=torch.bfloat16)

    out_anc_old = advance_local_window_kv(raw_model.blocks[9:], kv_ref[6:], h_w_anc, p_t, raw_model.cos, raw_model.sin, kv_slot=s_t)
    out_anc_new = advance_local_window_kv_fast(raw_model.blocks[9:], kv_fst[6:], h_w_anc, p_t, raw_model.cos, raw_model.sin, kv_slot=s_t)
    anc_cos = F.cosine_similarity(out_anc_old.float().reshape(1, -1), out_anc_new.float().reshape(1, -1)).item()
    assert anc_cos > 0.9999
    ms_anc_o = bench_ms(lambda: advance_local_window_kv(raw_model.blocks[9:], kv_ref[6:], h_w_anc, p_t, raw_model.cos, raw_model.sin, kv_slot=s_t), warmup=3, reps=6)
    ms_anc_n = bench_ms(lambda: advance_local_window_kv_fast(raw_model.blocks[9:], kv_fst[6:], h_w_anc, p_t, raw_model.cos, raw_model.sin, kv_slot=s_t), warmup=3, reps=6)

    out_ear_old = advance_local_window_kv(raw_model.blocks[3:9], kv_ref[:6], h_w_ear, p_t, raw_model.cos, raw_model.sin, kv_slot=s_t)
    out_ear_new = advance_local_window_kv_fast(raw_model.blocks[3:9], kv_fst[:6], h_w_ear, p_t, raw_model.cos, raw_model.sin, kv_slot=s_t)
    ear_cos = F.cosine_similarity(out_ear_old.float().reshape(1, -1), out_ear_new.float().reshape(1, -1)).item()
    assert ear_cos > 0.9999
    ms_ear_o = bench_ms(lambda: advance_local_window_kv(raw_model.blocks[3:9], kv_ref[:6], h_w_ear, p_t, raw_model.cos, raw_model.sin, kv_slot=s_t), warmup=3, reps=6)
    ms_ear_n = bench_ms(lambda: advance_local_window_kv_fast(raw_model.blocks[3:9], kv_fst[:6], h_w_ear, p_t, raw_model.cos, raw_model.sin, kv_slot=s_t), warmup=3, reps=6)

    print(f"[{tag} Anchor L09..12 W=4] cos={anc_cos:.6f} | old={ms_anc_o:6.2f}ms -> fast={ms_anc_n:6.2f}ms ({ms_anc_o/ms_anc_n:.2f}x)")
    print(f"[{tag} Early  L03..09 W=1] cos={ear_cos:.6f} | old={ms_ear_o:6.2f}ms -> fast={ms_ear_n:6.2f}ms ({ms_ear_o/ms_ear_n:.2f}x)")

# Time Readout topk(64) vs target_nll vs selected_log_probs at N=15729 to spot any remaining readout bottleneck:
x_ro = torch.randn((15729, 512), device=device, dtype=torch.bfloat16, requires_grad=True)
ms_topk = bench_ms(lambda: ctrl_t.lens_anchor.topk_pool_ids(x_ro, 64), warmup=2, reps=6)
pool_ids = ctrl_t.lens_anchor.topk_pool_ids(x_ro, 64)
from jlens_ablation import select_diverse_from_pool
ms_div = bench_ms(lambda: select_diverse_from_pool(pool_ids, ctrl_t.lens_anchor, k=8), warmup=2, reps=6)
ids8, _, _, _ = select_diverse_from_pool(pool_ids, ctrl_t.lens_anchor, k=8)
ms_nll = bench_ms(lambda: ctrl_t.readout.fast_nograd_target_nll(torch.randn((15729 * 40, 512), device=device, dtype=torch.bfloat16), torch.randint(0, 50256, (15729 * 40,), device=device)), warmup=2, reps=4)
ms_sel = bench_ms(lambda: (ctrl_t.lens_anchor.selected_log_probs(x_ro, ids8).sum().backward(), setattr(x_ro, 'grad', None)), warmup=2, reps=6)
print(f"[Readout Breakdown @ N=15729 (6%)] topk64={ms_topk:.2f}ms | diverse_k8={ms_div:.2f}ms | nograd_nll_629k_rows={ms_nll:.2f}ms | sel_logprob_fwd_bwd={ms_sel:.2f}ms")
logger.close({"kind": "ablation_bench", "linear_mode": "bf16", "jlens_mode": "ablation", "ms_per_step": round(ms_cur, 2)})
print(f"BENCHMARK COMPLETE -> {log_p}")
PY
