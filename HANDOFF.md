# Autonomous Handoff Brief — `jlens-loss-experiments` (Mac Mini `home` -> Featherless MI325X)

You are running on the always-on Mac mini (`chenguangs-Mac-mini.local`) inside `/Users/chen/Documents/GitHub/jlens-loss-experiments`.
Both `gh` (authenticated as `someoneexistsontheinternet` via `~/.config/gh/hosts.yml`) and `featherless-cloud` (`~/.local/bin/featherless-cloud`, authenticated via `~/.config/featherless-cloud.yaml`) are installed and ready.

---

## 1. Non-Negotiable Rules & Budgets

1. **ZERO SECRETS IN GIT OR JOB ARGS:**
   - NEVER commit `.env` files, API keys, tokens, or environment variable dumps to GitHub.
   - NEVER pass local environment variables via `--env` to `featherless-cloud exec` (the repo is public; containers clone over unauthenticated `https://github.com/someoneexistsontheinternet/jlens-loss-experiments.git`).
   - Always run `python3 tools/secret_scan.py` (automatically invoked by `tools/submit_mi325x.py`) before `git push`.
2. **ONE GPU PER JOB (`1x MI325X`):**
   - Every GPU Run Job MUST use `--gpu-type mi325x --gpu-count 1 --template rocm-pytorch --volume jlens-data:/data`.
   - You MAY run multiple `1x MI325X` jobs in parallel during sweeps to save wall-clock time, while strictly respecting total GPU-hour budgets!
3. **STRICT COMPUTE BUDGETS (`1x MI325X` = $3.29/GPU-hr):**
   - **Stage 1 — BF16 `70M` Kernel Tuning + Baseline Hparam Sweep:** **AT MOST `3.0` MI325X GPU-hours total.**
   - **Stage 2 — FP8 Block-Scaling Kernel Engineering & Ablations (`fp8_rowwise`, `mxfp8_e8m0`, `fp8_hier_bf16_fp32`):** **AT LEAST `1.0` and AT MOST `3.0` MI325X GPU-hours total.**
   - Track cumulative MI325X GPU-hours via `python3 tools/sync_logs.py` after every wave of jobs.
4. **IMMUTABLE PER-ATTEMPT SCRIPTS & ROLLBACK PROVENANCE:**
   - Every attempt gets its own script in `./scripts/NNN_<slug>.sh`.
   - Commit & `git push` BEFORE launching so `tools/submit_mi325x.py` pins the exact `COMMIT_SHA` (`git clone` -> `git checkout $COMMIT_SHA` inside the container).
   - Store **compact gzip-compressed JSONL** logs (`logs/<run_id>_<sha7>.jsonl.gz`) in GitHub + summary rows in `runs/manifest.jsonl` (not verbose human-readable walls of text).

---

## 2. Cloud Storage (`vol://jlens-data`, 50 GB)

- Volume `jlens-data` (`50 GB`, `< 100 GiB`) is already created and mounted at `/data` in jobs:
  - `/data/fineweb10B/`: Pre-tokenized GPT-2 `uint16` shards (`fineweb_val_000000.bin` `10.48M` tokens + `fineweb_train_000001..000040.bin` `4.0B` tokens + `READY.json`).
  - Check CPU staging job status first: `featherless-cloud runjob list` / `featherless-cloud volume ls vol://jlens-data/fineweb10B` (job `stage-fineweb10b-cpu` was kicked off on `16 vCPU` to populate this with zero GPU spend).
  - `/data/logs/`: Every job writes `<run_id>_<sha7>.jsonl.gz` here. Sync down anytime with `python3 tools/sync_logs.py`.

---

## 3. Execution Sequence

### Phase 0: Verify Dataset Volume Ready
1. Check `featherless-cloud runjob list` and `featherless-cloud volume ls vol://jlens-data/fineweb10B` until `READY.json` is present.

### Phase 1: BF16 `70M` Kernel Optimization & Baseline Hparam Sweep (`<= 3.0` MI325X GPU-hrs)
1. **Kernel & Compile Check (`./scripts/001_mi325x_kernel_bench.sh`, ~5-10 mins):**
   - Launch on `1x MI325X` via `python3 tools/submit_mi325x.py scripts/001_mi325x_kernel_bench.sh`.
   - Inspect ROCm version, `gfx942` arch, FlashAttention/SDPA speed, `torch.compile` speedup, and baseline BF16 step time on `71.38M` NanoGPT (`12L, D=512, 8Hx64, SwiGLU d_ff=1792`, `256`-aligned).
   - Tune BF16 kernel knobs (fused RoPE/RMSNorm, `torch.compile` mode, microbatch size `64x1024` or `128x1024` on `256 GB` HBM3e!) until BF16 MFU/tok/s is maximized.
2. **BF16 `70M` Hparam Sweep (`./scripts/010_*.sh .. ./scripts/019_*.sh`, stay strictly `<= 3.0` GPU-hrs cumulative):**
   - Sweep optimizer (`Muon+AdamW` vs pure `AdamW`), peak learning rates (e.g. Muon `0.01, 0.02, 0.03`; AdamW `1.5e-3, 3e-3, 4.5e-3`), weight decay (`0.01, 0.05, 0.10`), and global batch tokens (`262k` vs `524k`) on fixed token budgets (`e.g. 0.5B - 1.0B` FineWeb tokens per run; on MI325X a `0.5B`-token `70M` run takes only ~8-12 minutes!).
   - Sync `.jsonl.gz` logs, update `runs/manifest.jsonl`, commit & push, and lock in the winning BF16 `70M` configuration.

### Phase 2: MI325X FP8 Block-Scaling Engineering & Ablations (`1.0 - 3.0` MI325X GPU-hrs)
Remember the `gfx942` (CDNA3) hardware ground truth:
- MI325X uses `torch.float8_e4m3fnuz` / `torch.float8_e5m2fnuz` (`max=240.0`) on MFMA matrix cores (`2,614 TFLOPS` peak), without CDNA4's single-op `e8m0` MFMA instruction.
- Iterate on `src/fp8_kernels.py` (autotuning `BLOCK_M, BLOCK_N, BLOCK_K, waves_per_eu, matrix_instr_nonkdim = 16 or 32` on real `gfx942`!) and ablate all 3 FP8 families both in `bench_kernels.py` (TFLOPS + grad cosine SNR) and full `70M` training runs (`./scripts/020_*.sh ..`) at the locked best hparams:
  1. **`fp8_rowwise`:** `torch._scaled_mm` / rowwise FP8 baseline.
  2. **`mxfp8_e8m0`:** `float8_e4m3fnuz` weights & activations + `UE8M0` power-of-2 per-block scales (`1x32`, `1x64`, `1x128`) accumulated across MFMA K-tiles in FP32 VGPRs (also test Stochastic Rounding / RNE on `E8M0` scale exponent `ceil` vs `floor` to prevent clipping vs underflow!).
  3. **`fp8_hier_bf16_fp32` (NVFP4-inspired Two-Level FP8):** Fine `BF16` per-block scale (`1x64` or `1x128` act × `128x128` or `1x128` weight) multiplying partial MFMA tiles inside the K-loop, paired with outer `FP32` per-channel / per-tensor scales folded into the GEMM epilogue at zero inner-loop cost!
- Spend `1.0h - 3.0h` of MI325X compute tuning tile sizes, backward (`dgrad`/`wgrad`) transposed-quantization fusions, and running full head-to-head training curves against the locked BF16 baseline.

### Phase 3: `J-Lens` Loss Experiments (`./scripts/030_*.sh ..`)
- Once the `70M` baseline + FP8 kernel winner are locked and committed, run clean head-to-head `8-Horizon Focusing-Wave Stein J-Lens` (`--jlens-mode wave8`, already wired in `src/model_70m.py`) ablations on top of the locked baseline!
