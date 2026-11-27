# Phase 4 Handoff — Causal-Ablation Workspace `J-Lens` (`jlens_mode = "ablation"`)

## Context & Why We Are Skipping `Wave8` for Phase 4
In Phase 3, `Wave8` (`036` / `040`) achieved `-0.00374` / `-0.00360` nats *iso-token* over `019` (`3.66341`) / `039` (`3.66086`), **but**:
- Looking at the step curve (`st250: -0.0744 nats -> st500: -0.0159 -> st2000: -0.00374`), literal future-token supervision mostly front-loaded early trunk formation and then saturated/overfit once main CE entered the `log-log` power-law regime.
- At `+2.20%` extra FLOPs and `+10.2%` wall-clock (`465 ms` vs `413.7 ms`), `-0.0037` nats **narrowly lost strict Iso-FLOP (`~ -0.005` nats hurdle)** and **lost Iso-Wallclock (`~ -0.025` nats hurdle)**.

We have now implemented our new **Causal-Ablation Workspace `J-Lens`** in `src/jlens_ablation.py` (`--jlens-mode ablation` in `src/train.py`, with verification suite `tools/verify_jlens_ablation.py`).

---

## Architecture of `src/jlens_ablation.py`
Instead of guessing what surface tokens intermediate layers should represent, `CausalAblationJLensController` supervises intermediate layers as a **semantic / next-chunk workspace**:
1. **Diverse Active Hypotheses (`top-64 -> k in {8, 16}`):** Reads clean hidden state $h_{\ell, p}$ through a frozen `J-Lens` snapshot ($J_\ell$, running activation covariance $\Sigma_\ell$, and frozen `[out_norm, tok_emb]` softcapped readout) and greedily selects up to $k$ diverse active candidates (`|cos| <= 0.85`).
2. **Covariance-Matched Oblique Suppression (`--abl-metric cov` vs `euclidean`):** Converts read covector $w_i = U_i J_\ell$ into primal activation write vector $d_i = \Sigma_\ell w_i$, projecting out $w_i^\top x_{\text{ed}} = (1 - s) w_i^\top x$ without megaphone-neuron overshoot, plus **1 joint all-$k$ suppression probe** ($U_{\text{all}}$) via `(W^T D + eps I)^{-1} W^T x`.
3. **Exact KV-Cached `W`-Token Lookahead Window (`--abl-window 4`):** Reuses clean prefix `(K, V)` (`0 .. p - 1`) directly from stored `interm` states and advances all `k + 2` probes of length `W` (`[p .. p + W - 1]`) in parallel under `no_grad` (~250x fewer FLOPs than full-sequence cloning, verified bit-equivalent in `tools/verify_jlens_ablation.py`!).
4. **Layerwise Anchor Telescoping (`L03 -> L09 -> L12`):**
   - `anchor_layer = 9` runs `L09 -> L12` (`W`-token window) against exact downstream **gold NLL**.
   - `early_layer = 3` runs **only `L03 -> L09`** (`W=1` at `p`) against `L09`'s **hop-normalized** (`--abl-hop-norm 1`) fixed causal target (`score_at_anchor`).
5. **Superposition-Safe + Bounded Clean Loss (`--abl-loss-kind superpos_unlikelihood`) & `k + 1` Coalition Conservation (`--abl-coalition-norm 1`):**
   - Positive $u_j > 0$ removes sibling positive features from each feature's softmax denominator (`log(q_j / (q_j + q_bg))`) so co-active workspace concepts don't cannibalize each other.
   - Negative $u_j < 0$ uses bounded **unlikelihood** (`-|u_j| log(1 - q_j) >= 0`, vanishing grad as $q_j \to 0$).
6. **Delayed `log-log` Onset & Pre-Cooldown Anneal (`--jlens-start-frac 0.20 --jlens-end-frac 0.75`):** Stays completely OFF (`0` overhead) during initial steep syntax descent (`0..20%`), activates on the `log-log` curve, and anneals to `0` before WSD cooldown (`75%`).

---

## Experimental Mandate & Ablation Plan (Keep Total Phase 4 Budget `<= 6.0` MI325X GPU-hrs)

All runs use `scripts/050_*.sh ..` (keeps Stage 1 & Stage 2 budget buckets untouched in `tools/sync_logs.py`). Always commit & push immutable scripts before launching via `python3 tools/submit_mi325x.py`, and run up to 4–6 single-MI325X jobs in parallel per wave.

### Wave 4.0 — Unit Verification & Kernel Micro-Timing (`050_verify_and_bench_ablation.sh`, ~3 mins)
1. Run `tools/verify_jlens_ablation.py` on `1x MI325X` to confirm all 5 mathematical & KV-window equivalence checks pass on `gfx942`.
2. Benchmark step wall-clock (`ms/step`) & `extra_flops_pct` across `probe_seqs in {32, 64, 128, 256}`, `jlens_every in {1, 2, 4}`, `k in {8, 16}`, `window in {1, 4, 8}` (and optimize any hot PyTorch ops in `src/jlens_ablation.py` if wall-clock exceeds `~2x` theoretical FLOP overhead!).

### Wave 4.1 — Empirical Baseline Iso-FLOP / Iso-Wallclock Scaling Curve + Initial Weight Sweep
We MUST measure true **Iso-FLOP** and **Iso-Wallclock** directly rather than guessing slopes!
1. **Baseline Compute-Scaling Ruler (`bf16`, locked `019` hparams `muon_lr=0.025, adam_lr=0.008, wd=0.01`, Seed `42`):**
   - We already have `019` (`2000` steps = `0.524B` tok, `3.66341`, `873.8s`).
   - Run **`051_bf16_scale_2050st.sh`** (`+2.5%` FLOPs & time), **`052_bf16_scale_2200st.sh`** (`+10.0%` FLOPs & time), and **`053_bf16_scale_3000st.sh`** (`+50.0%` FLOPs & time = `0.786B` tokens). This pins down the **exact post-WSD-cooldown Pareto frontier `ValCE_BF16(FLOPs, WallTime)`**!
2. **Low-Tax Causal-Ablation Weight Sweep (`< 1–1.5%` FLOPs, `probe_seqs=64, k=8, W=4, every=2, start=0.20, end=0.75`):**
   - Sweep `--jlens-weight` across `{0.02, 0.05, 0.12, 0.25}` (`054`–`057`) at `2000` steps (`Seed 42`) to find the optimal causal-ablation weight `w*` and check if it beats the interpolated `ValCE_BF16(FLOPs, WallTime)` curve **both Iso-FLOP and Iso-Wallclock**!

### Wave 4.2 — Component Mechanism Ablations (at Winning `w*`)
Ablate each theoretical mechanism one-at-a-time against the Wave 4.1 winner:
1. **Lookahead Window (`--abl-window`):** `W = 1` (same-token only — zero KV cross-position lookahead credit!) vs `W = 4` vs `W = 8`.
2. **Loss Formulation (`--abl-loss-kind`):** `superpos_unlikelihood` (default) vs `unlikelihood` (categorical sibling competition) vs `signed_raw` (literal `-u_i log q_i` stability test).
3. **Covariance Metric & Conservation:** `--abl-metric euclidean` (raw `U_i @ J` without `Sigma_l`) vs `cov`, and `--abl-coalition-norm 0` / `--abl-hop-norm 0` / `--abl-enable-early 0` (anchor-only `L09` vs telescoped `L03 -> L09`).
4. **Onset Schedule (`--jlens-start-frac`):** `0.0` (from step 1 during fast unigram cliff) vs `0.20` (`log-log` curve onset).

### Wave 4.3 — Separate FLOP Scaling Dynamics (Probe-Compute Scaling vs Total-Compute Scaling!)
Explicitly study **both** FLOP-scaling axes separately:
1. **Axis A — Scaling Probe FLOPs Up at Fixed Data Tokens (`2000` steps = `0.524B` tokens):**
   - Scale probe compute up ~`16x` across 4 rungs:
     - Rung 1 (`~0.1%` extra FLOPs): `probe_seqs=32, k=8, every=4`
     - Rung 2 (`~0.4%` extra FLOPs): `probe_seqs=64, k=8, every=2`
     - Rung 3 (`~1.6%` extra FLOPs): `probe_seqs=128, k=8, every=1`
     - Rung 4 (`~5.5%` extra FLOPs): `probe_seqs=256, k=16, every=1`
   - Plot/tabulate both **raw `Val CE` vs Probe FLOPs** (measuring the sample-efficiency ceiling per token!) AND **Iso-FLOP net advantage vs `ValCE_BF16(FLOPs)`** (identifying the exact optimal probe-to-trunk FLOP ratio $\alpha^*$!).
2. **Axis B — Scaling Total Training Compute Up (`3000` steps = `0.786B` tokens, + Seed `137` Replication):**
   - Run the best Causal-Ablation `J-Lens` config at **`3000` steps** head-to-head against `053_bf16_scale_3000st.sh` (checking whether the causal-ablation advantage *grows* or *shrinks* with `1.5x` total compute, unlike `Wave8`!), PLUS a **Seed `137`** replication at `2000` steps against `039` (`3.66086`).

---

## Reporting
Sync all `.jsonl.gz` logs via `python3 tools/sync_logs.py`, update `README.md` and `experiment_log.md` with clear **Iso-Token, Iso-FLOP, Iso-Wallclock, Component Ablation, and Probe/Total FLOP-Scaling tables**, verify `python3 tools/secret_scan.py`, and push to `origin/main`.
