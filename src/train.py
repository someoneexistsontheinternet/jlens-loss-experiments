"""
Single-MI325X 70M NanoGPT Training & Sweep Runner:
  - Reads pre-tokenized FineWeb-10B `.bin` shards from `/data/fineweb10B`.
  - Supports Muon+AdamW or pure AdamW, configurable LR/WD/batch/steps/linear_mode, `torch.compile`, and 8-Horizon Wave `J-Lens`.
  - Evaluates exact FineWeb `fineweb_val_000000.bin` (`2.1M` streaming + `10.48M` final) and logs compact `.jsonl.gz`.
"""
import argparse
import math
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from model_70m import NanoGPT70MConfig, NanoGPT70M
from jlens_wave8 import Wave8SteinJLensController
from jlens_ablation import CausalAblationJLensController
from fast_kernels import compiled_newton_schulz_5, fast_clip_grad_norm_
from fp8_kernels import TORCH_FP8_DTYPE, FP8_MAX, IS_GFX942
from telemetry import CompressedJSONLLogger


def load_fineweb_shard(path: Path) -> np.memmap:
    hdr = np.fromfile(path, dtype=np.int32, count=256)
    assert hdr[0] == 20240520 and hdr[1] == 1, f"Bad shard header in {path}"
    return np.memmap(path, dtype=np.uint16, mode="r", offset=1024, shape=(int(hdr[2]),))


class ShardedBatchStream:
    def __init__(self, shards: list[np.memmap], B: int, T: int, seed: int = 42):
        self.shards = shards
        self.B, self.T = B, T
        self.span = B * T + 1
        self.rng  = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(shards))
        self.si, self.pos = 0, 0

    def next_batch(self, device: torch.device):
        sh = self.shards[self.order[self.si]]
        if self.pos + self.span > len(sh):
            self.si = (self.si + 1) % len(self.order)
            if self.si == 0:
                self.order = self.rng.permutation(len(self.shards))
            sh = self.shards[self.order[self.si]]
            self.pos = int(self.rng.integers(0, min(4096, max(1, len(sh) - self.span))))
        buf = torch.from_numpy(np.asarray(sh[self.pos : self.pos + self.span], dtype=np.int64)).to(device, non_blocking=True)
        self.pos += self.B * self.T
        return buf[:-1].view(self.B, self.T), buf[1:].view(self.B, self.T)


@torch.no_grad()
def newton_schulz_5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16)
    X = X / (X.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-7).to(torch.bfloat16))
    trans = X.size(-2) > X.size(-1)
    if trans:
        X = X.transpose(-2, -1)
    for _ in range(steps):
        A = torch.matmul(X, X.transpose(-2, -1))
        X = a * X + torch.matmul(b * A + c * torch.matmul(A, A), X)
    if trans:
        X = X.transpose(-2, -1)
    return X * (0.2 * math.sqrt(max(G.size(-2), G.size(-1))))


class HybridMuonAdamW:
    def __init__(self, model: nn.Module, opt_kind: str, muon_lr: float, adam_lr: float, wd: float):
        self.opt_kind = opt_kind
        self.base_muon_lr = muon_lr
        self.wd = wd
        self.muon_params = []
        adam_emb, adam_2d, adam_1d = [], [], []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if opt_kind == "muon" and p.ndim >= 2 and "tok_emb" not in name:
                self.muon_params.append(p)
            elif "tok_emb" in name:
                adam_emb.append(p)
            elif p.ndim >= 2:
                adam_2d.append(p)
            else:
                adam_1d.append(p)

        self.muon_bufs = [torch.zeros_like(p, dtype=torch.float32) for p in self.muon_params]
        self.muon_by_shape = {}
        for p, buf in zip(self.muon_params, self.muon_bufs):
            ps, bufs = self.muon_by_shape.setdefault(p.shape, ([], []))
            ps.append(p)
            bufs.append(buf)
        groups = [
            {"params": adam_emb, "lr": adam_lr * 0.8, "base_lr": adam_lr * 0.8, "weight_decay": 0.01},
            {"params": adam_2d,  "lr": adam_lr,       "base_lr": adam_lr,       "weight_decay": wd},
            {"params": adam_1d,  "lr": adam_lr,       "base_lr": adam_lr,       "weight_decay": 0.0},
        ]
        self.adam = torch.optim.AdamW([g for g in groups if g["params"]], betas=(0.9, 0.95), fused=torch.cuda.is_available())

    def zero_grad(self):
        for p in self.muon_params:
            p.grad = None
        self.adam.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self, lr_mult: float):
        for pg in self.adam.param_groups:
            pg["lr"] = pg["base_lr"] * lr_mult
        self.adam.step()

        if self.muon_params:
            lr_m = self.base_muon_lr * lr_mult
            ns_fn = compiled_newton_schulz_5 if self.muon_params[0].is_cuda else newton_schulz_5
            for _, (ps, bufs) in self.muon_by_shape.items():
                act = [(p, buf, p.grad.float()) for p, buf in zip(ps, bufs) if p.grad is not None]
                if not act:
                    continue
                p_act = [t[0] for t in act]
                b_act = [t[1] for t in act]
                g_act = [t[2] for t in act]
                torch._foreach_mul_(b_act, 0.95)
                torch._foreach_add_(b_act, g_act)
                nesterov = torch._foreach_add(g_act, b_act, alpha=0.95)
                u_stk = ns_fn(torch.stack(nesterov, dim=0))
                torch._foreach_mul_(p_act, 1.0 - lr_m * self.wd)
                torch._foreach_add_(p_act, [u.to(p_act[0].dtype) for u in u_stk.unbind(0)], alpha=-lr_m)


@torch.no_grad()
def evaluate_val(raw_model: NanoGPT70M, val_arr: np.memmap, eval_tokens: int, batch_size: int, device: torch.device) -> float:
    was_tr = raw_model.training
    raw_model.eval()
    T = raw_model.cfg.seq_len
    n_steps = min(eval_tokens, len(val_arr) - 1) // (batch_size * T)
    losses = []
    for i in range(n_steps):
        s = i * batch_size * T
        buf = torch.from_numpy(np.asarray(val_arr[s : s + batch_size * T + 1], dtype=np.int64)).to(device)
        vx, vy = buf[:-1].view(batch_size, T), buf[1:].view(batch_size, T)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = raw_model(vx, vy)
        losses.append(float(loss.item()))
    if was_tr:
        raw_model.train()
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_ROOT", "/data") + "/fineweb10B")
    ap.add_argument("--log-dir",  default=os.environ.get("DATA_ROOT", "/data") + "/logs")
    ap.add_argument("--linear-mode", default="bf16", choices=["bf16", "fp8_rowwise", "mxfp8_e8m0", "fp8_hier_bf16_fp32"])
    ap.add_argument("--fp8-block-k", type=int, default=128)
    ap.add_argument("--e8m0-mode", default="ceil_240", choices=["ceil_240", "ceil_pow2", "rne_240", "sr_240"])
    ap.add_argument("--hier-w-2d", type=int, default=0)
    ap.add_argument("--jlens-mode", default="none", choices=["none", "wave8", "ablation", "hybrid", "hybrid_wave8_abl"])
    ap.add_argument("--jlens-weight", type=float, default=0.35)
    ap.add_argument("--jlens-horizons", type=int, default=8)
    ap.add_argument("--jlens-start-frac", type=float, default=0.0)
    ap.add_argument("--jlens-end-frac", type=float, default=1.0)
    ap.add_argument("--jlens-every", type=int, default=1, help="Apply J-Lens every N steps (reduces FLOP/time tax)")
    ap.add_argument("--jlens-upd-every", type=int, default=25)
    # Hybrid Stage 1 (Low-FLOP Sparse Wave8 Multi-Horizon Lookahead Prior -> Stage 2 Causal-Ablation RL):
    ap.add_argument("--hybrid-w8-weight", type=float, default=0.05)
    ap.add_argument("--hybrid-w8-start-frac", type=float, default=0.0)
    ap.add_argument("--hybrid-w8-end-frac", type=float, default=0.55)
    ap.add_argument("--hybrid-w8-every", type=int, default=4)
    ap.add_argument("--hybrid-w8-layers", type=str, default="", help="Optional comma-separated 1-indexed taps e.g. '3,5,7,9'")
    ap.add_argument("--hybrid-w8-ema", type=float, default=0.90)
    # Phase 4 Causal-Ablation Workspace J-Lens flags:
    ap.add_argument("--abl-early-layer", type=int, default=3)
    ap.add_argument("--abl-anchor-layer", type=int, default=9)
    ap.add_argument("--abl-active-frac", type=float, default=0.02, help="Fraction of batch tokens probed (e.g. 0.005, 0.02, 0.06, 0.15; 0 uses --abl-probe-seqs)")
    ap.add_argument("--abl-probe-seqs", type=int, default=64)
    ap.add_argument("--abl-pool-size", type=int, default=64)
    ap.add_argument("--abl-k", type=int, default=8)
    ap.add_argument("--abl-window", type=int, default=4)
    ap.add_argument("--abl-max-cosine", type=float, default=0.85)
    ap.add_argument("--abl-select-mode", default="greedy_cos", choices=["greedy_cos", "maxmin", "kmeans", "emeans"])
    ap.add_argument("--abl-strength", type=float, default=1.0)
    ap.add_argument("--abl-match-norm", type=int, default=1)
    ap.add_argument("--abl-metric", default="cov", choices=["cov", "euclidean"])
    ap.add_argument("--abl-j-mode", default="ols_norm", choices=["ols_norm", "identity", "true_jacobian"])
    ap.add_argument("--abl-live-readout", type=int, default=0)
    ap.add_argument("--abl-reuse-clean-nll", type=int, default=0)
    ap.add_argument("--abl-jac-samples", type=int, default=16)
    ap.add_argument("--abl-loss-kind", default="superpos_unlikelihood", choices=["superpos_unlikelihood", "unlikelihood", "signed_raw", "exp_signed_clamped"])
    ap.add_argument("--abl-exp-beta", type=float, default=0.0)
    ap.add_argument("--abl-exp-norm", type=int, default=1)
    ap.add_argument("--abl-bystander-tau", type=float, default=0.0)
    ap.add_argument("--abl-bystander-w", type=float, default=0.0)
    ap.add_argument("--abl-bystander-unscaled", type=int, default=0)
    ap.add_argument("--abl-coalition-norm", type=int, default=1)
    ap.add_argument("--abl-gamma-max", type=float, default=4.0)
    ap.add_argument("--abl-hop-norm", type=int, default=1)
    ap.add_argument("--abl-enable-early", type=int, default=1)
    ap.add_argument("--abl-layer-mode", default="fixed", choices=["fixed", "random_gap4", "random_anc_7_10", "split_6_9"])
    ap.add_argument("--abl-pos-mode", default="hard_easy", choices=["hard_easy", "uniform"])
    ap.add_argument("--abl-nll-cap", type=float, default=0.0)
    ap.add_argument("--abl-live-target-head", type=int, default=0)
    ap.add_argument("--abl-gold-w", type=float, default=0.0, help="Zero-extra-FLOP Gold SFT weight inside existing selected_log_probs")
    ap.add_argument("--abl-shared-w8-weight", type=float, default=0.0, help="Shared-position 8-horizon lookahead weight on active L09 tokens")
    ap.add_argument("--abl-shared-w8-kvocab", type=int, default=1536)
    ap.add_argument("--abl-detach-prev", type=int, default=0)
    ap.add_argument("--abl-ema", type=float, default=0.50)
    ap.add_argument("--opt", default="muon", choices=["muon", "adamw"])
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--adam-lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--micro-batch", type=int, default=128)
    ap.add_argument("--batch-tokens", type=int, default=262144)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--compile-mode", default="default")
    ap.add_argument("--ce-backend", default="triton", choices=["triton", "eager"])
    ap.add_argument("--ce-chunk-sz", type=int, default=16384)
    ap.add_argument("--fused-block-ops", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.cache_size_limit = 64

    data_dir = Path(args.data_dir)
    val_arr  = load_fineweb_shard(data_dir / "fineweb_val_000000.bin")
    tr_paths = sorted(data_dir.glob("fineweb_train_*.bin"))
    assert tr_paths, f"No fineweb_train_*.bin found in {data_dir}"
    tr_shards = [load_fineweb_shard(p) for p in tr_paths]

    cfg = NanoGPT70MConfig(
        seq_len=args.seq_len,
        linear_mode=args.linear_mode,
        fp8_block_k=args.fp8_block_k,
        e8m0_mode=args.e8m0_mode,
        hier_w_2d=bool(args.hier_w_2d),
        jlens_mode=args.jlens_mode,
        jlens_weight=args.jlens_weight,
        ce_backend=args.ce_backend,
        ce_chunk_sz=args.ce_chunk_sz,
        fused_block_ops=bool(args.fused_block_ops),
    )
    raw_model = NanoGPT70M(cfg).to(device)
    n_tot, n_trunk, flops_tok = raw_model.count_params_and_flops()
    fwd_model = torch.compile(raw_model, mode=args.compile_mode) if args.compile else raw_model

    is_hybrid = args.jlens_mode in ("hybrid", "hybrid_wave8_abl")
    use_jlens = args.jlens_mode in ("wave8", "ablation") or is_hybrid
    w8_tap_layers = [int(x) for x in args.hybrid_w8_layers.split(",") if x.strip()] if args.hybrid_w8_layers.strip() else None
    jlens_ctrl = None
    w8_ctrl = None
    if args.jlens_mode == "wave8" or is_hybrid:
        w8_ctrl = Wave8SteinJLensController(
            n_layers=cfg.n_layers,
            d_model=cfg.d_model,
            vocab_pad=cfg.vocab_pad,
            micro_tokens=args.micro_batch * args.seq_len,
            device=device,
            unigram_path=data_dir / "unigram_counts_50432.npy",
            horizons=args.jlens_horizons,
            ema=args.hybrid_w8_ema if is_hybrid else 0.90,
            softcap=cfg.softcap,
            layer_indices=w8_tap_layers if is_hybrid else None,
        )
        if args.jlens_mode == "wave8":
            jlens_ctrl = w8_ctrl
    if args.jlens_mode == "ablation" or is_hybrid:
        jlens_ctrl = CausalAblationJLensController(
            raw_model=raw_model,
            device=device,
            early_layer=args.abl_early_layer,
            anchor_layer=args.abl_anchor_layer,
            active_frac=args.abl_active_frac,
            probe_seqs=args.abl_probe_seqs,
            pool_size=args.abl_pool_size,
            k_dirs=args.abl_k,
            window=args.abl_window,
            max_cosine=args.abl_max_cosine,
            select_mode=args.abl_select_mode,
            strength=args.abl_strength,
            match_norm=bool(args.abl_match_norm),
            metric=args.abl_metric,
            j_mode=args.abl_j_mode,
            live_readout=bool(args.abl_live_readout),
            reuse_clean_nll=bool(args.abl_reuse_clean_nll),
            loss_kind=args.abl_loss_kind,
            exp_beta=args.abl_exp_beta,
            exp_norm=bool(args.abl_exp_norm),
            bystander_tau=args.abl_bystander_tau,
            bystander_w=args.abl_bystander_w,
            bystander_unscaled=bool(args.abl_bystander_unscaled),
            coalition_norm=bool(args.abl_coalition_norm),
            gamma_max=args.abl_gamma_max,
            hop_norm=bool(args.abl_hop_norm),
            enable_early=bool(args.abl_enable_early),
            layer_mode=args.abl_layer_mode,
            pos_mode=args.abl_pos_mode,
            nll_cap=args.abl_nll_cap,
            live_target_head=bool(args.abl_live_target_head),
            gold_w=args.abl_gold_w,
            shared_w8_weight=args.abl_shared_w8_weight,
            shared_w8_horizons=args.jlens_horizons,
            shared_w8_k_vocab=args.abl_shared_w8_kvocab,
            unigram_path=data_dir / "unigram_counts_50432.npy",
            jac_samples=args.abl_jac_samples,
            detach_prev=bool(args.abl_detach_prev),
            ema=args.abl_ema,
            seed=args.seed,
        )

    grad_accum = max(1, args.batch_tokens // (args.micro_batch * args.seq_len))
    toks_step  = args.micro_batch * args.seq_len * grad_accum
    opt = HybridMuonAdamW(raw_model, args.opt, args.muon_lr, args.adam_lr, args.wd)
    stream = ShardedBatchStream(tr_shards, args.micro_batch, args.seq_len, seed=args.seed)

    run_id = os.environ.get("RUN_ID", f"train_{args.linear_mode}_{args.jlens_mode}")
    sha7   = os.environ.get("GIT_COMMIT_SHA", "local")[:7]
    log_p  = Path(args.log_dir) / f"{run_id}_{sha7}.jsonl.gz"
    abl_meta = {}
    if args.jlens_mode == "ablation" or is_hybrid:
        if args.abl_layer_mode == "random_gap4":
            layers_desc = "random_gap4(2..6->6..10)"
        elif args.abl_layer_mode == "random_anc_7_10":
            layers_desc = "rand_anc(7..10)" if not args.abl_enable_early else f"{args.abl_early_layer}->rand(7..10)"
        elif args.abl_layer_mode == "split_6_9":
            layers_desc = "split_anc(6..9)"
        else:
            layers_desc = f"{args.abl_early_layer}->{args.abl_anchor_layer}" if args.abl_enable_early else f"anc{args.abl_anchor_layer}"
        abl_meta = {
            "abl_layers": layers_desc,
            "abl_layer_mode": args.abl_layer_mode,
            "abl_active_frac": args.abl_active_frac,
            "abl_probe_seqs": args.abl_probe_seqs,
            "abl_pool_size": args.abl_pool_size,
            "abl_k": args.abl_k,
            "abl_window": args.abl_window,
            "abl_select_mode": args.abl_select_mode,
            "abl_metric": args.abl_metric,
            "abl_j_mode": args.abl_j_mode,
            "abl_live_readout": args.abl_live_readout,
            "abl_reuse_clean_nll": args.abl_reuse_clean_nll,
            "abl_loss_kind": args.abl_loss_kind,
            "abl_exp_beta": args.abl_exp_beta,
            "abl_bystander_tau": args.abl_bystander_tau,
            "abl_bystander_w": args.abl_bystander_w,
            "abl_bystander_unscaled": args.abl_bystander_unscaled,
            "abl_gamma_max": args.abl_gamma_max,
            "abl_nll_cap": args.abl_nll_cap,
            "abl_live_target_head": args.abl_live_target_head,
            "abl_gold_w": args.abl_gold_w,
            "abl_shared_w8_weight": args.abl_shared_w8_weight,
            "abl_coalition_norm": args.abl_coalition_norm,
            "abl_hop_norm": args.abl_hop_norm,
            "jlens_every": args.jlens_every,
        }
        if is_hybrid:
            abl_meta.update({
                "hybrid_w8_weight": args.hybrid_w8_weight,
                "hybrid_w8_start_frac": args.hybrid_w8_start_frac,
                "hybrid_w8_end_frac": args.hybrid_w8_end_frac,
                "hybrid_w8_every": args.hybrid_w8_every,
                "hybrid_w8_layers": w8_ctrl.tap_layers,
            })
    logger = CompressedJSONLLogger(log_p, {
        "kind": "train_70m",
        "n_params": n_tot,
        "n_trunk": n_trunk,
        "flops_per_tok": flops_tok,
        "linear_mode": args.linear_mode,
        "fp8_block_k": args.fp8_block_k,
        "jlens_mode": args.jlens_mode,
        "jlens_weight": args.jlens_weight if use_jlens else 0.0,
        "jlens_horizons": args.jlens_horizons if (args.jlens_mode == "wave8" or is_hybrid) else 0,
        "jlens_start_frac": args.jlens_start_frac if use_jlens else 0.0,
        "jlens_end_frac": args.jlens_end_frac if use_jlens else 1.0,
        "opt": args.opt,
        "lr": args.muon_lr if args.opt == "muon" else args.adam_lr,
        "adam_lr": args.adam_lr,
        "wd": args.wd,
        "toks_per_step": toks_step,
        "steps_target": args.steps,
        "seed": args.seed,
        "fp8_spec": f"{TORCH_FP8_DTYPE}(max={FP8_MAX},gfx942={IS_GFX942})",
        **abl_meta,
    })

    stable_end = int(args.steps * 0.80)
    j_start    = int(args.steps * args.jlens_start_frac)
    j_end      = int(args.steps * args.jlens_end_frac)
    w8_start   = int(args.steps * args.hybrid_w8_start_frac)
    w8_end     = int(args.steps * args.hybrid_w8_end_frac)
    t_win = time.perf_counter()
    last_s = 0
    steady_ms = []
    extra_jlens_flops_tot = 0.0
    jlens_active_steps = 0
    last_jlens_upd_step = -10**9
    last_w8_upd_step    = -10**9

    for step in range(1, args.steps + 1):
        if step < args.warmup:
            lr_m, j_cos = step / max(1, args.warmup), 1.0
        elif step <= stable_end:
            lr_m, j_cos = 1.0, 1.0
        else:
            w_cos = 0.5 * (1.0 + math.cos(math.pi * (step - stable_end) / max(1, args.steps - stable_end)))
            lr_m  = 0.05 + 0.95 * w_cos
            j_cos = w_cos

        ramp_steps = max(50, int(round(0.025 * args.steps)))
        if args.jlens_start_frac > 0.0 or args.jlens_end_frac < 1.0:
            eff_start = max(args.warmup if args.jlens_start_frac == 0.0 else j_start, 0)
            if step < j_start or step >= j_end:
                j_cos = 0.0
            elif args.jlens_start_frac > 0.0 and step < j_start + ramp_steps:
                # Smooth 2.5%-horizon ramp-in once entering log-log regime (50 steps at 2000st, 380 steps at 15200st)
                j_cos = 0.5 * (1.0 - math.cos(math.pi * (step - j_start + 1) / float(ramp_steps)))
            elif step <= eff_start:
                j_cos = 1.0
            else:
                j_cos = 0.5 * (1.0 + math.cos(math.pi * (step - eff_start) / max(1, j_end - eff_start)))

        w8_cos = 0.0
        if is_hybrid:
            eff_w8_start = max(args.warmup if args.hybrid_w8_start_frac == 0.0 else w8_start, 0)
            if step < w8_start or step >= w8_end:
                w8_cos = 0.0
            elif args.hybrid_w8_start_frac > 0.0 and step < w8_start + ramp_steps:
                w8_cos = 0.5 * (1.0 - math.cos(math.pi * (step - w8_start + 1) / float(ramp_steps)))
            elif step <= eff_w8_start:
                w8_cos = 1.0
            else:
                w8_cos = 0.5 * (1.0 + math.cos(math.pi * (step - eff_w8_start) / max(1, w8_end - eff_w8_start)))

        cadence_ok   = ((step - max(1, j_start)) % max(1, args.jlens_every) == 0)
        active_jlens = use_jlens and (j_cos > 1e-6) and cadence_ok
        cadence_w8   = (step == 1) or ((step % max(1, args.hybrid_w8_every)) == 0)
        active_w8    = is_hybrid and (w8_cos > 1e-6) and cadence_w8
        prewarm_snap = (
            (args.jlens_mode == "ablation" or is_hybrid)
            and j_start > 0
            and step in (max(1, j_start - args.jlens_upd_every), max(1, j_start - 1))
        )

        opt.zero_grad()
        ce_sum, jl_sum, w8_sum = 0.0, 0.0, 0.0
        abl_stats_last = None
        need_w8_cand = (active_jlens and args.jlens_mode == "wave8") or active_w8
        cand_ids = w8_ctrl.sample_vocab_candidates() if need_w8_cand else None
        if active_jlens or active_w8:
            jlens_active_steps += 1

        for micro_i in range(grad_accum):
            xb, yb = stream.next_batch(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if is_hybrid and (active_w8 or active_jlens):
                    ce_loss, tok_nll, interm, x_raw = fwd_model(xb, yb, return_interm=True)
                    interm_det = None
                    tot_loss = ce_loss
                    if active_w8:
                        did_w8_ref = (micro_i == 0) and (step == 1 or (step - last_w8_upd_step) >= args.jlens_upd_every)
                        if did_w8_ref:
                            interm_det = [t.detach() for t in interm]
                            w8_ctrl.refresh_lenses(interm_det, x_raw, yb)
                            last_w8_upd_step = step
                        w8_loss, w8_obs = w8_ctrl.compute_loss(
                            interm, raw_model.out_norm.weight, raw_model.tok_emb.weight, yb, tok_nll, cand_ids
                        )
                        anchor = (ce_loss.detach() / w8_obs.clamp_min(1e-4)).clamp(0.20, 1.0)
                        tot_loss = tot_loss + (args.hybrid_w8_weight * w8_cos * anchor) * w8_loss
                        w8_sum += float(w8_obs.item()) / grad_accum
                        extra_jlens_flops_tot += w8_ctrl.estimate_call_flops(did_refresh=did_w8_ref)
                    if active_jlens:
                        did_abl_ref = (micro_i == 0) and (not jlens_ctrl._initialized or (step - last_jlens_upd_step) >= args.jlens_upd_every)
                        if did_abl_ref:
                            if interm_det is None:
                                interm_det = [t.detach() for t in interm]
                            jlens_ctrl.refresh_snapshot(raw_model, interm_det, x_raw, yb=yb)
                            last_jlens_upd_step = step
                        j_loss, j_obs, abl_stats_last = jlens_ctrl.compute_loss(raw_model, interm, yb, tok_nll)
                        tot_loss = tot_loss + (args.jlens_weight * j_cos) * j_loss
                        jl_sum += float(j_obs.item()) / grad_accum
                        extra_jlens_flops_tot += jlens_ctrl.estimate_call_flops(
                            args.micro_batch,
                            abl_stats_last["n_act"],
                            abl_stats_last["n_seqs"],
                            abl_stats_last["l_ear"],
                            abl_stats_last["l_anc"],
                        )
                    elif prewarm_snap and micro_i == 0:
                        if interm_det is None:
                            interm_det = [t.detach() for t in interm]
                        jlens_ctrl.refresh_snapshot(raw_model, interm_det, x_raw, yb=yb)
                        last_jlens_upd_step = step
                elif active_jlens and args.jlens_mode == "wave8":
                    ce_loss, tok_nll, interm, x_raw = fwd_model(xb, yb, return_interm=True)
                    did_w8_ref = (micro_i == 0) and (step == 1 or (step - last_jlens_upd_step) >= args.jlens_upd_every)
                    if did_w8_ref:
                        jlens_ctrl.refresh_lenses([t.detach() for t in interm], x_raw, yb)
                        last_jlens_upd_step = step
                    j_loss, j_obs = jlens_ctrl.compute_loss(
                        interm, raw_model.out_norm.weight, raw_model.tok_emb.weight, yb, tok_nll, cand_ids
                    )
                    anchor = (ce_loss.detach() / j_obs.clamp_min(1e-4)).clamp(0.20, 1.0)
                    tot_loss = ce_loss + (args.jlens_weight * j_cos * anchor) * j_loss
                    jl_sum += float(j_obs.item()) / grad_accum
                    extra_jlens_flops_tot += jlens_ctrl.estimate_call_flops(did_refresh=did_w8_ref)
                elif active_jlens and args.jlens_mode == "ablation":
                    ce_loss, tok_nll, interm, x_raw = fwd_model(xb, yb, return_interm=True)
                    did_abl_ref = (micro_i == 0) and (not jlens_ctrl._initialized or (step - last_jlens_upd_step) >= args.jlens_upd_every)
                    if did_abl_ref:
                        jlens_ctrl.refresh_snapshot(raw_model, [t.detach() for t in interm], x_raw, yb=yb)
                        last_jlens_upd_step = step
                    j_loss, j_obs, abl_stats_last = jlens_ctrl.compute_loss(raw_model, interm, yb, tok_nll)
                    tot_loss = ce_loss + (args.jlens_weight * j_cos) * j_loss
                    jl_sum += float(j_obs.item()) / grad_accum
                    extra_jlens_flops_tot += jlens_ctrl.estimate_call_flops(
                        args.micro_batch,
                        abl_stats_last["n_act"],
                        abl_stats_last["n_seqs"],
                        abl_stats_last["l_ear"],
                        abl_stats_last["l_anc"],
                        did_refresh=did_abl_ref,
                    )
                elif prewarm_snap and micro_i == 0:
                    ce_loss, _, interm, x_raw = fwd_model(xb, yb, return_interm=True)
                    jlens_ctrl.refresh_snapshot(raw_model, [t.detach() for t in interm], x_raw, yb=yb)
                    last_jlens_upd_step = step
                    tot_loss = ce_loss
                else:
                    ce_loss, _ = fwd_model(xb, yb, return_interm=False)
                    tot_loss = ce_loss
            (tot_loss / grad_accum).backward()
            ce_sum += float(ce_loss.detach().item()) / grad_accum

        gnorm = fast_clip_grad_norm_(raw_model.parameters(), 1.0)
        opt.step(lr_m)

        do_eval = (step % args.eval_every == 0) or (step == args.steps)
        do_log  = (step == 1) or (step % args.log_every == 0) or do_eval
        if do_log:
            torch.cuda.synchronize()
            dt    = time.perf_counter() - t_win
            ms_st = (dt * 1000.0) / max(1, step - last_s)
            if step > args.warmup:
                steady_ms.append(ms_st)
            tok_s  = toks_step / max(ms_st / 1000.0, 1e-6)
            tflops = (tok_s * flops_tok) / 1e12

            rec = {
                "type": "step", "step": step,
                "ce": round(ce_sum, 4), "gnorm": round(gnorm, 3), "lr_m": round(lr_m, 3),
                "ms_per_step": round(ms_st, 2), "tok_per_s": round(tok_s, 1), "tflops": round(tflops, 1),
            }
            if use_jlens:
                rec["jlens_ce"] = round(jl_sum, 4)
                rec["j_cos"] = round(j_cos, 3)
            if is_hybrid:
                rec["w8_ce"] = round(w8_sum, 4)
                rec["w8_cos"] = round(w8_cos, 3)
            if abl_stats_last is not None:
                rec.update(abl_stats_last)
            if do_eval:
                v_ce = evaluate_val(raw_model, val_arr, eval_tokens=2_097_152, batch_size=min(64, args.micro_batch), device=device)
                rec["val_ce"] = round(v_ce, 4)
            logger.write(rec)
            v_str = f" | val={rec['val_ce']:.4f}" if "val_ce" in rec else ""
            w8_str = f" | w8={w8_sum:.3f}({w8_cos:.2f})" if is_hybrid and w8_cos > 1e-6 else ""
            j_str = f" | jce={jl_sum:.4f}" if use_jlens else ""
            u_str = f" | |uA|={abl_stats_last['u_anc_abs']:.4f}" if abl_stats_last is not None else ""
            print(f"[{run_id}] st {step:4d}/{args.steps} | ce={ce_sum:.4f}{w8_str}{j_str}{u_str}{v_str} | gn={gnorm:.2f} | {ms_st:.1f}ms/st ({tok_s/1e3:.0f}k tok/s, {tflops:.0f} TF)", flush=True)
            t_win = time.perf_counter(); last_s = step

    full_val = evaluate_val(raw_model, val_arr, eval_tokens=10_485_760, batch_size=min(64, args.micro_batch), device=device)
    med_ms   = float(np.median(steady_ms)) if steady_ms else ms_st
    mean_ms  = float(np.mean(steady_ms)) if steady_ms else ms_st
    med_toks = toks_step / max(med_ms / 1000.0, 1e-6)
    base_tot_flops  = float(args.steps) * float(toks_step) * float(flops_tok)
    extra_flops_pct = 100.0 * (extra_jlens_flops_tot / max(base_tot_flops, 1.0))
    logger.close({
        "kind": "train_70m",
        "linear_mode": args.linear_mode,
        "jlens_mode": args.jlens_mode,
        "steps": args.steps,
        "jlens_active_steps": jlens_active_steps,
        "extra_flops_pct": round(extra_flops_pct, 3),
        "val_ce": round(full_val, 5),
        "val_ppl": round(math.exp(min(full_val, 20.0)), 3),
        "ms_per_step": round(med_ms, 2),
        "mean_ms_per_step": round(mean_ms, 2),
        "tok_per_s": round(med_toks, 1),
        "total_tokens": args.steps * toks_step,
    })
    print(f"=== DONE {run_id} | Full 10.5M Val CE = {full_val:.5f} | ExtraFLOPs = +{extra_flops_pct:.2f}% | Mean {mean_ms:.1f} ms/st (Med {med_ms:.1f}) -> {log_p} ===", flush=True)


if __name__ == "__main__":
    main()
