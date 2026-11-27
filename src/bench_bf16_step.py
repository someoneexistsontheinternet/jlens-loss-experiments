"""
Phase 1 BF16 70M End-to-End Step Profiler & Kernel Tuner on MI325X (gfx942):
  1. Verifies numerical exactness of FusedTritonChunkedSoftcapCEFn vs eager ChunkedSoftcapCEFn (loss diff + dh/dw cosine).
  2. Checks SDPA FlashAttention backend availability on ROCm PyTorch 2.12.
  3. Benchmarks full 262,144-token training step (Fwd + Bwd + grad_clip + Muon/AdamW step) across:
       - Baseline: eager CE (chunk=8192) + compile=0 vs compile=1, micro_batch=128
       - Fused Triton CE (chunk=32768) + FastRMSNorm + compile=1 (default vs max-autotune-no-cudagraphs)
       - Microbatch = 64 (ga=4) vs 128 (ga=2) vs 256 (ga=1) on 256 GB HBM3e
"""
import argparse
import os
import time
from pathlib import Path
import torch
import torch.nn.functional as F

from model_70m import NanoGPT70MConfig, NanoGPT70M, ChunkedSoftcapCEFn, FusedTritonChunkedSoftcapCEFn
from train import HybridMuonAdamW
from telemetry import CompressedJSONLLogger


def verify_triton_ce(device: torch.device) -> dict:
    torch.manual_seed(123)
    M, D, V = 16384, 512, 50304
    h_ref = torch.randn((M, D), device=device, dtype=torch.bfloat16, requires_grad=True)
    w_ref = (torch.randn((V, D), device=device, dtype=torch.float32) * 0.02).requires_grad_(True)
    tgt   = torch.randint(0, 50257, (M,), device=device, dtype=torch.int64)

    loss_e, nll_e = ChunkedSoftcapCEFn.apply(h_ref, w_ref, tgt, 30.0, 8192)
    dh_e, dw_e = torch.autograd.grad(loss_e, (h_ref, w_ref))

    h_tri = h_ref.detach().clone().requires_grad_(True)
    w_tri = w_ref.detach().clone().requires_grad_(True)
    loss_t, nll_t = FusedTritonChunkedSoftcapCEFn.apply(h_tri, w_tri, tgt, 30.0, 16384)
    dh_t, dw_t = torch.autograd.grad(loss_t, (h_tri, w_tri))

    loss_diff = abs(float(loss_t.item()) - float(loss_e.item()))
    nll_max_diff = float((nll_t - nll_e).abs().max().item())
    dh_cos = float(F.cosine_similarity(dh_t.float().reshape(1, -1), dh_e.float().reshape(1, -1)).item())
    dw_cos = float(F.cosine_similarity(dw_t.float().reshape(1, -1), dw_e.float().reshape(1, -1)).item())
    assert loss_diff < 5e-3 and dh_cos > 0.9999 and dw_cos > 0.9999, (
        f"Triton CE mismatch! loss_diff={loss_diff}, dh_cos={dh_cos}, dw_cos={dw_cos}"
    )
    return {
        "type": "verify_ce",
        "loss_eager": round(float(loss_e.item()), 6),
        "loss_triton": round(float(loss_t.item()), 6),
        "loss_abs_diff": round(loss_diff, 6),
        "nll_max_diff": round(nll_max_diff, 5),
        "dh_cos": round(dh_cos, 6),
        "dw_cos": round(dw_cos, 6),
    }


def bench_config(
    label: str,
    ce_backend: str,
    ce_chunk_sz: int,
    do_compile: bool,
    compile_mode: str,
    micro_batch: int,
    batch_tokens: int = 262144,
    warmup_steps: int = 3,
    timed_steps: int = 8,
) -> dict:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(42)

    seq_len = 1024
    grad_accum = batch_tokens // (micro_batch * seq_len)
    cfg = NanoGPT70MConfig(
        seq_len=seq_len,
        linear_mode="bf16",
        ce_backend=ce_backend,
        ce_chunk_sz=ce_chunk_sz,
    )
    raw_model = NanoGPT70M(cfg).to(device)
    _, _, flops_tok = raw_model.count_params_and_flops()
    fwd_model = torch.compile(raw_model, mode=compile_mode) if do_compile else raw_model
    opt = HybridMuonAdamW(raw_model, "muon", 0.02, 3e-3, 0.01)

    xb = torch.randint(0, 50257, (micro_batch, seq_len), device=device, dtype=torch.int64)
    yb = torch.randint(0, 50257, (micro_batch, seq_len), device=device, dtype=torch.int64)

    def run_step():
        opt.zero_grad()
        for _ in range(grad_accum):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = fwd_model(xb, yb, return_interm=False)
            (loss / grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        opt.step(1.0)

    for _ in range(warmup_steps):
        run_step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(timed_steps):
        run_step()
    torch.cuda.synchronize()
    ms_step = ((time.perf_counter() - t0) * 1000.0) / timed_steps
    tok_s   = batch_tokens / (ms_step / 1000.0)
    tflops  = (tok_s * flops_tok) / 1e12
    mfu_pct = (tflops / 1307.0) * 100.0
    mem_gb  = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    del fwd_model, raw_model, opt, xb, yb
    torch.cuda.empty_cache()

    return {
        "type": "bf16_step_bench",
        "label": label,
        "ce_backend": ce_backend,
        "ce_chunk_sz": ce_chunk_sz,
        "compile": int(do_compile),
        "compile_mode": compile_mode if do_compile else "none",
        "micro_batch": micro_batch,
        "grad_accum": grad_accum,
        "ms_per_step": round(ms_step, 2),
        "tok_per_s": round(tok_s, 1),
        "tflops": round(tflops, 1),
        "mfu_pct": round(mfu_pct, 2),
        "peak_mem_gb": round(mem_gb, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=os.environ.get("DATA_ROOT", "/data") + "/logs")
    args = ap.parse_args()

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    run_id = os.environ.get("RUN_ID", "002_bf16_step_tune")
    sha7   = os.environ.get("GIT_COMMIT_SHA", "local")[:7]
    out_p  = Path(args.log_dir) / f"{run_id}_{sha7}.jsonl.gz"

    logger = CompressedJSONLLogger(out_p, {
        "kind": "kernel_bench",
        "linear_mode": "bf16",
        "gcn_arch": getattr(props, "gcnArchName", "unknown"),
        "multi_processor_count": props.multi_processor_count,
        "total_mem_gb": round(props.total_memory / (1024 ** 3), 1),
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
    })

    print(f"=== MI325X BF16 70M Step Tuning | Arch={getattr(props, 'gcnArchName', '')} CUs={props.multi_processor_count} HBM={props.total_memory/(1024**3):.1f}GB ===")
    ce_ver = verify_triton_ce(device)
    logger.write(ce_ver)
    print(f"  [CE Verify] loss_diff={ce_ver['loss_abs_diff']:.6f} nll_max_diff={ce_ver['nll_max_diff']:.5f} dh_cos={ce_ver['dh_cos']:.6f} dw_cos={ce_ver['dw_cos']:.6f}", flush=True)

    matrix = [
        ("eager_ce_nocompile_mb128",    "eager",  8192,  False, "default", 128),
        ("eager_ce_compile_mb128",      "eager",  8192,  True,  "default", 128),
        ("triton_ce_nocompile_mb128",   "triton", 32768, False, "default", 128),
        ("triton_ce_compile_mb64",      "triton", 32768, True,  "default", 64),
        ("triton_ce_compile_mb128",     "triton", 32768, True,  "default", 128),
        ("triton_ce_compile_mb256",     "triton", 32768, True,  "default", 256),
        ("triton_ce_maxauto_mb128",     "triton", 32768, True,  "max-autotune-no-cudagraphs", 128),
        ("triton_ce_maxauto_mb256",     "triton", 32768, True,  "max-autotune-no-cudagraphs", 256),
    ]

    best = None
    for cfg_row in matrix:
        try:
            rec = bench_config(*cfg_row)
            logger.write(rec)
            print(f"  {rec['label']:28s} | {rec['ms_per_step']:6.1f} ms/st | {rec['tok_per_s']/1e3:7.1f}k tok/s | "
                  f"{rec['tflops']:5.1f} TF ({rec['mfu_pct']:5.1f}% MFU) | mem={rec['peak_mem_gb']:5.1f} GB", flush=True)
            if best is None or rec["tok_per_s"] > best["tok_per_s"]:
                best = rec
        except Exception as e:
            print(f"  {cfg_row[0]:28s} | FAILED: {e}", flush=True)
            logger.write({"type": "error", "label": cfg_row[0], "err": str(e)})

    summary = {
        "kind": "kernel_bench",
        "linear_mode": "bf16",
        "best_label": best["label"] if best else "none",
        "ms_per_step": best["ms_per_step"] if best else None,
        "tok_per_s": best["tok_per_s"] if best else None,
    }
    logger.close(summary)
    print(f"=== WINNER: {summary['best_label']} -> {summary['ms_per_step']} ms/st ({summary['tok_per_s']/1e3:.1f}k tok/s) ===", flush=True)


if __name__ == "__main__":
    main()
