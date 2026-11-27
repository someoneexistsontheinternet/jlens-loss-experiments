"""
MI325X (gfx942) Kernel Benchmark & Numerical Fidelity Suite v2:
  1. Benchmarks Fwd + dgrad + wgrad TFLOPS and FP32-reference cosine/SNR across exact 70M NanoGPT GEMM shapes,
     including all 3 FP8 families (`fp8_rowwise`, `mxfp8_e8m0` x {ceil_pow2, ceil_240, rne_240, sr_240}, `fp8_hier_bf16_fp32` 1D & 2D).
  2. Optionally (`--bench-model 1`) benchmarks full 70M NanoGPT end-to-end training step speed across FP8 families with `compile=1`.
"""
import argparse
import os
import time
from pathlib import Path
import torch
import torch.nn.functional as F

from fp8_kernels import QuantLinear, TORCH_FP8_DTYPE, FP8_MAX, IS_GFX942
from telemetry import CompressedJSONLLogger


SHAPES_70M = [
    ("attn_qkv", 65536, 512, 1536),
    ("attn_out", 65536, 512, 512),
    ("mlp_gu",   65536, 512, 3584),
    ("mlp_down", 65536, 1792, 512),
]

MODES = [
    ("bf16",               128, "ceil_240",  False),
    ("fp8_rowwise",        128, "ceil_240",  False),
    ("mxfp8_e8m0",         64,  "ceil_pow2", False),
    ("mxfp8_e8m0",         32,  "ceil_240",  False),
    ("mxfp8_e8m0",         64,  "ceil_240",  False),
    ("mxfp8_e8m0",         128, "ceil_240",  False),
    ("mxfp8_e8m0",         64,  "rne_240",   False),
    ("mxfp8_e8m0",         64,  "sr_240",    False),
    ("fp8_hier_bf16_fp32", 64,  "ceil_240",  False),
    ("fp8_hier_bf16_fp32", 128, "ceil_240",  False),
    ("fp8_hier_bf16_fp32", 128, "ceil_240",  True),
]


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    af, bf = a.float().reshape(-1), b.float().reshape(-1)
    return float(F.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item())


def bench_one(shape_name: str, M: int, K: int, N: int, mode: str, block_k: int, e8m0_mode: str, hier_2d: bool, reps: int = 12):
    device = torch.device("cuda")
    torch.manual_seed(42)
    x_ref = (torch.randn((M, K), device=device, dtype=torch.float32) * (1.0 + 3.0 * (torch.rand((1, K), device=device) > 0.95).float())).requires_grad_(True)
    w_ref = (torch.randn((N, K), device=device, dtype=torch.float32) / (K ** 0.5)).requires_grad_(True)
    dy    = torch.randn((M, N), device=device, dtype=torch.bfloat16)

    y_ref = F.linear(x_ref, w_ref)
    gx_ref, gw_ref = torch.autograd.grad((y_ref * dy.float()).sum(), (x_ref, w_ref))

    lin = QuantLinear(K, N, mode=mode, block_k=block_k, e8m0_mode=e8m0_mode, hier_w_2d=hier_2d).to(device)
    with torch.no_grad():
        lin.weight.copy_(w_ref.to(torch.bfloat16))
    x_bf = x_ref.detach().to(torch.bfloat16).requires_grad_(True)

    y_q = lin(x_bf)
    gx_q, gw_q = torch.autograd.grad((y_q.float() * dy.float()).sum(), (x_bf, lin.weight))
    fwd_cos = cos_sim(y_q, y_ref)
    dx_cos  = cos_sim(gx_q, gx_ref)
    dw_cos  = cos_sim(gw_q, gw_ref)
    rel_err = float((y_q.detach().float() - y_ref.detach()).norm() / y_ref.detach().norm())

    for _ in range(3):
        x_bf.grad = lin.weight.grad = None
        y_t = lin(x_bf)
        y_t.backward(dy)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(reps):
        x_bf.grad = lin.weight.grad = None
        y_t = lin(x_bf)
        y_t.backward(dy)
    torch.cuda.synchronize()
    ms = ((time.perf_counter() - t0) * 1000.0) / reps
    flops_tot = 6.0 * M * K * N
    tflops = (flops_tot / (ms / 1000.0)) / 1e12

    return {
        "type": "bench",
        "shape": shape_name, "M": M, "K": K, "N": N,
        "mode": mode, "block_k": block_k, "e8m0_mode": e8m0_mode, "hier_2d": hier_2d,
        "ms_fwdbwd": round(ms, 3),
        "tflops_6n": round(tflops, 1),
        "rel_err": round(rel_err, 5),
        "fwd_cos": round(fwd_cos, 6),
        "dx_cos": round(dx_cos, 6),
        "dw_cos": round(dw_cos, 6),
    }


def bench_full_70m_steps(logger):
    from model_70m import NanoGPT70MConfig, NanoGPT70M
    from train import HybridMuonAdamW

    device = torch.device("cuda")
    step_modes = [
        ("bf16_compile_mb256",               "bf16",               128, "ceil_240", False, True),
        ("fp8_rowwise_compile_mb256",        "fp8_rowwise",        128, "ceil_240", False, True),
        ("mxfp8_b64_ceil240_nocompile",      "mxfp8_e8m0",         64,  "ceil_240", False, False),
        ("mxfp8_b64_ceil240_compile_mb256",  "mxfp8_e8m0",         64,  "ceil_240", False, True),
        ("mxfp8_b128_ceil240_compile_mb256", "mxfp8_e8m0",         128, "ceil_240", False, True),
        ("hier_b64_compile_mb256",           "fp8_hier_bf16_fp32", 64,  "ceil_240", False, True),
        ("hier_b128_compile_mb256",          "fp8_hier_bf16_fp32", 128, "ceil_240", False, True),
        ("hier_b128_2dw_compile_mb256",      "fp8_hier_bf16_fp32", 128, "ceil_240", True,  True),
    ]
    print("\n=== Full 70M NanoGPT End-to-End Step Shootout (262,144 tok/step, mb=256) ===", flush=True)
    for label, lmode, bk, emode, h2d, comp in step_modes:
        torch._dynamo.reset()
        torch._dynamo.config.cache_size_limit = 64
        torch.cuda.empty_cache()
        cfg = NanoGPT70MConfig(linear_mode=lmode, fp8_block_k=bk, e8m0_mode=emode, hier_w_2d=h2d)
        raw_m = NanoGPT70M(cfg).to(device)
        _, _, flops_tok = raw_m.count_params_and_flops()
        fwd_m = torch.compile(raw_m) if comp else raw_m
        opt = HybridMuonAdamW(raw_m, "muon", 0.025, 0.008, 0.01)
        xb = torch.randint(0, 50257, (256, 1024), device=device)
        yb = torch.randint(0, 50257, (256, 1024), device=device)

        for _ in range(3):
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = fwd_m(xb, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_m.parameters(), 1.0)
            opt.step(1.0)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(6):
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = fwd_m(xb, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_m.parameters(), 1.0)
            opt.step(1.0)
        torch.cuda.synchronize()
        ms = ((time.perf_counter() - t0) * 1000.0) / 6.0
        tok_s = 262144.0 / (ms / 1000.0)
        tf = (tok_s * flops_tok) / 1e12
        rec = {
            "type": "model_step_bench",
            "label": label, "mode": lmode, "block_k": bk, "e8m0_mode": emode, "hier_2d": h2d, "compile": int(comp),
            "ms_per_step": round(ms, 2), "tok_per_s": round(tok_s, 1), "tflops": round(tf, 1),
        }
        logger.write(rec)
        print(f"  {label:34s} : {ms:6.1f} ms/st | {tok_s/1e3:7.1f}k tok/s | {tf:5.1f} TF", flush=True)
        del fwd_m, raw_m, opt, xb, yb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=os.environ.get("DATA_ROOT", ".") + "/logs")
    ap.add_argument("--bench-model", type=int, default=0)
    args = ap.parse_args()

    run_id = os.environ.get("RUN_ID", "023_fp8_kernel_bench")
    sha7   = os.environ.get("GIT_COMMIT_SHA", "local")[:7]
    out_p  = Path(args.log_dir) / f"{run_id}_{sha7}.jsonl.gz"

    logger = CompressedJSONLLogger(out_p, {
        "kind": "kernel_bench",
        "linear_mode": "fp8_shootout_v2",
        "fp8_dtype": str(TORCH_FP8_DTYPE),
        "fp8_max": FP8_MAX,
        "is_gfx942": IS_GFX942,
    })

    print(f"=== MI325X Kernel Shootout v2 | FP8={TORCH_FP8_DTYPE} (max={FP8_MAX}, gfx942={IS_GFX942}) ===")
    rows = []
    for sname, M, K, N in SHAPES_70M:
        for mode, bk, emode, h2d in MODES:
            try:
                rec = bench_one(sname, M, K, N, mode, bk, emode, h2d)
                logger.write(rec)
                rows.append(rec)
                if mode == "mxfp8_e8m0":
                    tag = f"mxfp8(b={bk},{emode})"
                elif mode == "fp8_hier_bf16_fp32":
                    tag = f"hier(b={bk},{'2Dw' if h2d else '1D'})"
                else:
                    tag = mode
                print(f"  {sname:10s} ({M}x{K}x{N}) | {tag:28s} : {rec['ms_fwdbwd']:6.2f} ms ({rec['tflops_6n']:6.1f} TF) | "
                      f"err={rec['rel_err']:.4f} cos(y/dx/dw)={rec['fwd_cos']:.5f}/{rec['dx_cos']:.5f}/{rec['dw_cos']:.5f}", flush=True)
            except Exception as e:
                print(f"  {sname:10s} | {mode}(b={bk},{emode}) FAILED: {e}", flush=True)
                logger.write({"type": "error", "shape": sname, "mode": mode, "block_k": bk, "err": str(e)})

    if args.bench_model:
        bench_full_70m_steps(logger)

    logger.close({"n_benchmarks": len(rows), "kind": "kernel_bench", "linear_mode": "fp8_shootout_v2"})
    print(f"Saved compressed benchmark telemetry -> {out_p}")


if __name__ == "__main__":
    main()
