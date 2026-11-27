"""
MI325X (gfx942) Kernel Benchmark & Numerical Fidelity Suite:
  Benchmarks Fwd + dgrad + wgrad TFLOPS and FP32-reference cosine/SNR across exact 70M NanoGPT GEMM shapes!
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
    ("bf16",               128),
    ("fp8_rowwise",        128),
    ("mxfp8_e8m0",         32),
    ("mxfp8_e8m0",         64),
    ("mxfp8_e8m0",         128),
    ("fp8_hier_bf16_fp32", 64),
    ("fp8_hier_bf16_fp32", 128),
]


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    af, bf = a.float().reshape(-1), b.float().reshape(-1)
    return float(F.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item())


def bench_one(shape_name: str, M: int, K: int, N: int, mode: str, block_k: int, reps: int = 12):
    device = torch.device("cuda")
    torch.manual_seed(42)
    # Create heavy-tailed realistic activation + weight distribution
    x_ref = (torch.randn((M, K), device=device, dtype=torch.float32) * (1.0 + 3.0 * (torch.rand((1, K), device=device) > 0.95).float())).requires_grad_(True)
    w_ref = (torch.randn((N, K), device=device, dtype=torch.float32) / (K ** 0.5)).requires_grad_(True)
    dy    = torch.randn((M, N), device=device, dtype=torch.bfloat16)

    y_ref = F.linear(x_ref, w_ref)
    gx_ref, gw_ref = torch.autograd.grad((y_ref * dy.float()).sum(), (x_ref, w_ref))

    lin = QuantLinear(K, N, mode=mode, block_k=block_k).to(device)
    with torch.no_grad():
        lin.weight.copy_(w_ref.to(torch.bfloat16))
    x_bf = x_ref.detach().to(torch.bfloat16).requires_grad_(True)

    # Accuracy pass
    y_q = lin(x_bf)
    gx_q, gw_q = torch.autograd.grad((y_q.float() * dy.float()).sum(), (x_bf, lin.weight))
    fwd_cos = cos_sim(y_q, y_ref)
    dx_cos  = cos_sim(gx_q, gx_ref)
    dw_cos  = cos_sim(gw_q, gw_ref)
    rel_err = float((y_q.float() - y_ref).norm() / y_ref.norm())

    # Timing Fwd+Bwd
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
        "mode": mode, "block_k": block_k,
        "ms_fwdbwd": round(ms, 3),
        "tflops_6n": round(tflops, 1),
        "rel_err": round(rel_err, 5),
        "fwd_cos": round(fwd_cos, 6),
        "dx_cos": round(dx_cos, 6),
        "dw_cos": round(dw_cos, 6),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=os.environ.get("DATA_ROOT", ".") + "/logs")
    args = ap.parse_args()

    run_id = os.environ.get("RUN_ID", "001_kernel_bench")
    sha7   = os.environ.get("GIT_COMMIT_SHA", "local")[:7]
    out_p  = Path(args.log_dir) / f"{run_id}_{sha7}.jsonl.gz"

    logger = CompressedJSONLLogger(out_p, {
        "kind": "kernel_bench",
        "linear_mode": "all_fp8_vs_bf16",
        "fp8_dtype": str(TORCH_FP8_DTYPE),
        "fp8_max": FP8_MAX,
        "is_gfx942": IS_GFX942,
    })

    print(f"=== MI325X Kernel Shootout | FP8={TORCH_FP8_DTYPE} (max={FP8_MAX}, gfx942={IS_GFX942}) ===")
    rows = []
    for sname, M, K, N in SHAPES_70M:
        for mode, bk in MODES:
            try:
                rec = bench_one(sname, M, K, N, mode, bk)
                logger.write(rec)
                rows.append(rec)
                tag = f"{mode}(b={bk})" if "fp8" in mode and "row" not in mode else mode
                print(f"  {sname:10s} ({M}x{K}x{N}) | {tag:26s} : {rec['ms_fwdbwd']:6.2f} ms ({rec['tflops_6n']:6.1f} TF) | "
                      f"err={rec['rel_err']:.4f} cos(y/dx/dw)={rec['fwd_cos']:.5f}/{rec['dx_cos']:.5f}/{rec['dw_cos']:.5f}", flush=True)
            except Exception as e:
                print(f"  {sname:10s} | {mode}(b={bk}) FAILED: {e}", flush=True)
                logger.write({"type": "error", "shape": sname, "mode": mode, "block_k": bk, "err": str(e)})

    logger.close({"n_benchmarks": len(rows), "kind": "kernel_bench"})
    print(f"Saved compressed benchmark telemetry -> {out_p}")


if __name__ == "__main__":
    main()
