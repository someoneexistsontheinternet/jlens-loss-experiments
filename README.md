# jlens-loss-experiments

Systematic single-GPU **AMD Instinct MI325X (`gfx942`, ROCm)** kernel, precision, baseline-sweep, and **Focusing-Wave Stein `J-Lens`** training experiments at `~70M` NanoGPT scale on FineWeb-10B.

## Repository Layout
- `src/fp8_kernels.py` — MI325X (`gfx942` `float8_e4m3fnuz` & OCP `float8_e4m3fn`) linear backends:
  - `bf16`: Standard BF16 `rocBLAS / hipBLASLt` baseline.
  - `fp8_rowwise`: Rowwise/tensorwise FP8 via `_scaled_mm` + fused Triton quantizer.
  - `mxfp8_e8m0`: Block-scaled FP8 + `UE8M0` power-of-2 exponent scales (`1x32` / `1x64` / `1x128`) via custom ROCm Triton tiled-MFMA kernels.
  - `fp8_hier_bf16_fp32`: **NVFP4-inspired Two-Level FP8** — fine `BF16` per-block scales inside the MFMA K-loop paired with outer `FP32` per-channel / per-tensor scales folded into the GEMM epilogue.
- `src/bench_kernels.py` — Standalone MI325X kernel TFLOPS (`fwd`, `dgrad`, `wgrad`) + quantization SNR & gradient-cosine benchmark suite.
- `src/model_70m.py` — `71.38M`-param (`12L, D=512, 8H×64, SwiGLU d_ff=1792`, `256`-aligned) modern NanoGPT (RoPE, QK-RMSNorm, chunked logit-softcapped CE, optional 8-Horizon Wave `J-Lens`).
- `src/train.py` — Single-MI325X training & evaluation harness writing compact `.jsonl.gz` telemetry.
- `scripts/` — One immutable script per experimental attempt (`NNN_<slug>.sh`).
- `tools/submit_mi325x.py` — Pre-flight secret scanner + commit-pinned Featherless Cloud MI325X job launcher.
- `tools/sync_logs.py` — Syncs `/data/logs/*.jsonl.gz` from `vol://jlens-data`, updates `runs/manifest.jsonl`, and tracks MI325X GPU-hour budgets.
- `logs/*.jsonl.gz` & `runs/manifest.jsonl` — Compressed machine-readable run logs and commit-SHA ledger.
