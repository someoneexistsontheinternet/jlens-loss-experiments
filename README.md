# jlens-loss-experiments

Systematic single-GPU **AMD Instinct MI325X (`gfx942` CDNA3, 304 CUs, 256 GB HBM3e, ROCm 7.14 / PyTorch 2.12 / Triton 3.7.1)** kernel engineering, FP8 block-scaling ablations, baseline hyperparameter sweeps, and **8-Horizon Focusing-Wave Stein `J-Lens`** experiments at `71.38M` NanoGPT scale (`12L, D=512, 8H×64, SwiGLU d_ff=1792`, `256`-aligned) on FineWeb-10B (`0.524B` tokens/run, evaluated on full `10.48M`-token `fineweb_val_000000.bin`).

## Headline Results Summary

| Phase / Experiment | Run Script | Precision / Kernel | `J-Lens` Config | Full `10.48M` Val CE | Val PPL | Throughput (`tok/s`) |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| **Stage 1 — Pure AdamW Best** | `scripts/013_bf16_adamw_lr3e3_70m.sh` | `bf16` | `none` | `3.79002` | `44.26` | `638.6k` |
| **Stage 1 — Initial Muon Baseline** | `scripts/010_bf16_70m_baseline.sh` | `bf16` | `none` | `3.67654` | `39.51` | `624.4k` |
| **Stage 1 — Locked BF16 Champion (s42)** | `scripts/019_bf16_muon_emblr8e3_70m.sh` | `bf16` | `none` | `3.66341` | `38.99` | `633.7k` |
| **Stage 1 — Locked BF16 Control (s137)** | `scripts/039_bf16_locked_seed137_70m.sh` | `bf16` | `none` | `3.66086` | `38.90` | `634.4k` |
| **Stage 2 — MXFP8 `1x32` (`ceil_240`)** | `scripts/028_mxfp8_b32_ceil240_70m.sh` | `mxfp8_e8m0 (1x32)` | `none` | `3.66500` | `39.06` | `341.2k` |
| **Stage 2 — Two-Level HierFP8 `1x64`** | `scripts/027_fp8_hier_b64_1d_70m.sh` | `fp8_hier_bf16_fp32 (1x64)` | `none` | `3.66574` | `39.09` | `418.8k` |
| **Stage 2 — MXFP8 `1x128` (`ceil_240`)** | `scripts/029_mxfp8_b128_ceil240_70m.sh` | `mxfp8_e8m0 (1x128)` | `none` | `3.66602` | `39.10` | `470.4k` |
| **Stage 2 — Rowwise FP8 (`_scaled_mm`)** | `scripts/024_fp8_rowwise_70m_locked.sh` | `fp8_rowwise` | `none` | `3.66737` | `39.15` | `571.0k` |
| **Stage 2 — MXFP8 `1x64` (`rne_240`)** | `scripts/029b_mxfp8_b64_rne240_70m.sh` | `mxfp8_e8m0 (1x64, RNE)` | `none` | `3.79603` | `44.52` | `425.8k` |
| **Phase 3 — Uncalibrated `Wave8` (`w=0.35`)** | `scripts/031_jlens_wave8_locked_w035_70m.sh` | `bf16` | `wave8 (w=0.35, ef=1.0)` | `3.68704` | `39.93` | `564.9k` |
| **Phase 3 — `h=1` Control (`w=0.05, ef=0.75`)** | `scripts/037_jlens_h1_calibrated_anneal_70m.sh` | `bf16` | `h=1 (w=0.05, ef=0.75)` | `3.66356` | `39.00` | `589.6k` |
| **Phase 3 — Calibrated `Wave8` (s42)** | `scripts/036_jlens_wave8_calibrated_anneal_70m.sh` | `bf16` | `wave8 (w=0.05, ef=0.75)` | **`3.65967`** | **`38.85`** | `563.9k` |
| **Phase 3 — Calibrated `Wave8` (s137)** | `scripts/040_jlens_wave8_cal_seed137_70m.sh` | `bf16` | `wave8 (w=0.05, ef=0.75)` | **`3.65726`** | **`38.76`** | `563.6k` |
| **Phase 3 — Rowwise FP8 + `Wave8`** | `scripts/041_jlens_wave8_cal_fp8_rowwise_70m.sh` | `fp8_rowwise` | `wave8 (w=0.035, ef=0.65)` | **`3.66674`** | **`39.12`** | `516.7k` |

### Key Technical Takeaways
1. **BF16 `70M` Kernel & Optimizer Engineering (`Stage 1: 2.750 / 3.000` MI325X GPU-hrs):**
   - Replacing eager multi-pass softcap Cross-Entropy (`~240 GB` HBM traffic) with a **fused in-place 2-pass Triton online-softmax Softcap-CE kernel** + `FastRMSNorm` + `torch.compile` sped up `71.38M` training **`2.53x`** (`256.8k -> 649.5k tok/s`, `loss_diff = 0.000000`).
   - `Muon + AdamW` outperformed pure `AdamW` by **`-0.1135` nats** (`3.6765` vs `3.7900`), and raising tied-embedding/1D AdamW LR (`0.008`) with `muon_lr=0.025` lowered BF16 Val CE to **`3.66341`** (`seed=42`) / **`3.66086`** (`seed=137`).
2. **`gfx942` (`float8_e4m3fnuz`, `max=240.0`) FP8 Block-Scaling (`Stage 2: 2.594 / 3.000` MI325X GPU-hrs):**
   - Fused 2D tile transpose-quantize (`dgrad`/`wgrad`), L2 `GROUP_M` swizzling, `kpack=2` CDNA3 MFMA configs, and deterministic **Split-K `wgrad`** boosted `fp8_rowwise` **`4.37x`** (`453.2 TFLOPS`), `mxfp8_e8m0` **`2.14–3.34x`** (`289.5 TFLOPS`), and `fp8_hier_bf16_fp32` **`2.08–3.33x`** (`260.0 TFLOPS`).
   - Zero-clipping `UE8M0` exponent ceiling (`ceil_240`) beat RNE exponent rounding (`rne_240`) by **`-0.1282` nats** (`3.6678` vs `3.7960`), because rounding exponents down on floating-point `e4m3fnuz` clips dominant block outliers without adding mantissa precision.
   - NVFP4-inspired **Two-Level `BF16+FP32` (`fp8_hier_bf16_fp32`, `1x64`)** achieved the highest single-GEMM numerical fidelity (`rel_err=0.0331`, `dw_cos=0.99938`) and beat `1x64` `mxfp8_e8m0` in full training (`3.66574` vs `3.66781`), while `1x32` `mxfp8_e8m0` (`3.66500`) came within **`+0.00159` nats** of BF16.
3. **8-Horizon Focusing-Wave Stein `J-Lens` (`Phase 3: 3.432` MI325X GPU-hrs):**
   - Uncalibrated `w in {0.20, 0.35, 0.50}` degraded `k=1` CE linearly (`+0.010` to `+0.041` nats) because subsampling `15.6%` hardest/easiest tokens inflated gradient norms `~3.5x` inside Muon's scale-invariant `NS5` update through WSD cooldown.
   - Calibrating `w = 0.05` (`~0.35 × k_tot/M`) and cosine-annealing `J-Lens` to `0.0` at `75%` of steps (`before` WSD cooldown) yielded a **two-seed replicated win over BF16**: **`-0.00374` nats on Seed 42 (`3.66341 -> 3.65967`)** and **`-0.00360` nats on Seed 137 (`3.66086 -> 3.65726`)**, while a matched `h=1` single-horizon Stein control (`037`, `3.66356`) gave zero gain — isolating the improvement specifically to the **8-horizon focusing wave**.

## Repository Layout
- `src/fp8_kernels.py` — MI325X (`gfx942` `float8_e4m3fnuz`) linear backends (`bf16`, `fp8_rowwise`, `mxfp8_e8m0`, `fp8_hier_bf16_fp32`) with fused 2D transpose-quantize and deterministic Split-K `wgrad`.
- `src/model_70m.py` — `71.38M` NanoGPT with fused in-place Triton Softcap-CE and `FastRMSNorm`.
- `src/jlens_wave8.py` — 8-Horizon Focusing-Wave Stein `J-Lens` controller with Inductor-fused importance-sampled softcap loss.
- `src/train.py` — Single-MI325X training & evaluation runner outputting `.jsonl.gz` telemetry.
- `scripts/001_*.sh .. scripts/041_*.sh` — Immutable commit-pinned attempt scripts.
- `logs/*.jsonl.gz` & `runs/manifest.jsonl` — All 27 synced compressed telemetry logs and GPU-hour ledger.
