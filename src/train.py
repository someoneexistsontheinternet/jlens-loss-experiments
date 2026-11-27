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
        groups = [
            {"params": adam_emb, "lr": adam_lr * 0.8, "base_lr": adam_lr * 0.8, "weight_decay": 0.01},
            {"params": adam_2d,  "lr": adam_lr,       "base_lr": adam_lr,       "weight_decay": wd},
            {"params": adam_1d,  "lr": adam_lr,       "base_lr": adam_lr,       "weight_decay": 0.0},
        ]
        self.adam = torch.optim.AdamW([g for g in groups if g["params"]], betas=(0.9, 0.95))

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
            by_shape = {}
            for p, buf in zip(self.muon_params, self.muon_bufs):
                if p.grad is None:
                    continue
                g = p.grad.float()
                buf.mul_(0.95).add_(g)
                by_shape.setdefault(p.shape, []).append((p, g + 0.95 * buf))
            for _, items in by_shape.items():
                g_stk = torch.stack([it[1] for it in items], dim=0)
                u_stk = newton_schulz_5(g_stk)
                for idx_i, (p, _) in enumerate(items):
                    p.mul_(1.0 - lr_m * self.wd).add_(u_stk[idx_i].to(p.dtype), alpha=-lr_m)


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
    ap.add_argument("--jlens-mode", default="none", choices=["none", "wave8"])
    ap.add_argument("--jlens-weight", type=float, default=0.35)
    ap.add_argument("--jlens-horizons", type=int, default=8)
    ap.add_argument("--jlens-end-frac", type=float, default=1.0)
    ap.add_argument("--jlens-upd-every", type=int, default=25)
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
    ap.add_argument("--ce-chunk-sz", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
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
    )
    raw_model = NanoGPT70M(cfg).to(device)
    n_tot, n_trunk, flops_tok = raw_model.count_params_and_flops()
    fwd_model = torch.compile(raw_model, mode=args.compile_mode) if args.compile else raw_model

    use_jlens = (args.jlens_mode == "wave8")
    jlens_ctrl = None
    if use_jlens:
        jlens_ctrl = Wave8SteinJLensController(
            n_layers=cfg.n_layers,
            d_model=cfg.d_model,
            vocab_pad=cfg.vocab_pad,
            micro_tokens=args.micro_batch * args.seq_len,
            device=device,
            unigram_path=data_dir / "unigram_counts_50432.npy",
            horizons=args.jlens_horizons,
            softcap=cfg.softcap,
        )

    grad_accum = max(1, args.batch_tokens // (args.micro_batch * args.seq_len))
    toks_step  = args.micro_batch * args.seq_len * grad_accum
    opt = HybridMuonAdamW(raw_model, args.opt, args.muon_lr, args.adam_lr, args.wd)
    stream = ShardedBatchStream(tr_shards, args.micro_batch, args.seq_len, seed=args.seed)

    run_id = os.environ.get("RUN_ID", f"train_{args.linear_mode}_{args.jlens_mode}")
    sha7   = os.environ.get("GIT_COMMIT_SHA", "local")[:7]
    log_p  = Path(args.log_dir) / f"{run_id}_{sha7}.jsonl.gz"
    logger = CompressedJSONLLogger(log_p, {
        "kind": "train_70m",
        "n_params": n_tot,
        "n_trunk": n_trunk,
        "flops_per_tok": flops_tok,
        "linear_mode": args.linear_mode,
        "fp8_block_k": args.fp8_block_k,
        "jlens_mode": args.jlens_mode,
        "jlens_weight": args.jlens_weight if use_jlens else 0.0,
        "jlens_horizons": args.jlens_horizons if use_jlens else 0,
        "jlens_end_frac": args.jlens_end_frac if use_jlens else 1.0,
        "opt": args.opt,
        "lr": args.muon_lr if args.opt == "muon" else args.adam_lr,
        "adam_lr": args.adam_lr,
        "wd": args.wd,
        "toks_per_step": toks_step,
        "steps_target": args.steps,
        "fp8_spec": f"{TORCH_FP8_DTYPE}(max={FP8_MAX},gfx942={IS_GFX942})",
    })

    stable_end = int(args.steps * 0.80)
    t_win = time.perf_counter()
    last_s = 0
    steady_ms = []

    for step in range(1, args.steps + 1):
        if step < args.warmup:
            lr_m, j_cos = step / max(1, args.warmup), 1.0
        elif step <= stable_end:
            lr_m, j_cos = 1.0, 1.0
        else:
            w_cos = 0.5 * (1.0 + math.cos(math.pi * (step - stable_end) / max(1, args.steps - stable_end)))
            lr_m  = 0.05 + 0.95 * w_cos
            j_cos = w_cos

        if args.jlens_end_frac < 1.0:
            j_end = int(args.steps * args.jlens_end_frac)
            if step <= args.warmup:
                j_cos = 1.0
            elif step >= j_end:
                j_cos = 0.0
            else:
                j_cos = 0.5 * (1.0 + math.cos(math.pi * (step - args.warmup) / max(1, j_end - args.warmup)))

        active_jlens = use_jlens and (j_cos > 1e-6)
        opt.zero_grad()
        ce_sum, jl_sum = 0.0, 0.0
        cand_ids = jlens_ctrl.sample_vocab_candidates() if active_jlens else None

        for micro_i in range(grad_accum):
            xb, yb = stream.next_batch(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if active_jlens:
                    ce_loss, tok_nll, interm, x_raw = fwd_model(xb, yb, return_interm=True)
                    if micro_i == 0 and (step == 1 or step % args.jlens_upd_every == 0):
                        jlens_ctrl.refresh_lenses([t.detach() for t in interm], x_raw, yb)
                    j_loss, j_obs = jlens_ctrl.compute_loss(
                        interm, raw_model.out_norm.weight, raw_model.tok_emb.weight, yb, tok_nll, cand_ids
                    )
                    anchor = (ce_loss.detach() / j_obs.clamp_min(1e-4)).clamp(0.20, 1.0)
                    tot_loss = ce_loss + (args.jlens_weight * j_cos * anchor) * j_loss
                    jl_sum += float(j_obs.item()) / grad_accum
                else:
                    ce_loss, _ = fwd_model(xb, yb, return_interm=False)
                    tot_loss = ce_loss
            (tot_loss / grad_accum).backward()
            ce_sum += float(ce_loss.detach().item()) / grad_accum

        gnorm = float(torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0).item())
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
            if do_eval:
                v_ce = evaluate_val(raw_model, val_arr, eval_tokens=2_097_152, batch_size=min(64, args.micro_batch), device=device)
                rec["val_ce"] = round(v_ce, 4)
            logger.write(rec)
            v_str = f" | val={rec['val_ce']:.4f}" if "val_ce" in rec else ""
            j_str = f" | jce={jl_sum:.3f}" if use_jlens else ""
            print(f"[{run_id}] st {step:4d}/{args.steps} | ce={ce_sum:.4f}{j_str}{v_str} | gn={gnorm:.2f} | {ms_st:.1f}ms/st ({tok_s/1e3:.0f}k tok/s, {tflops:.0f} TF)", flush=True)
            t_win = time.perf_counter(); last_s = step

    full_val = evaluate_val(raw_model, val_arr, eval_tokens=10_485_760, batch_size=min(64, args.micro_batch), device=device)
    med_ms   = float(np.median(steady_ms)) if steady_ms else ms_st
    med_toks = toks_step / max(med_ms / 1000.0, 1e-6)
    logger.close({
        "kind": "train_70m",
        "linear_mode": args.linear_mode,
        "jlens_mode": args.jlens_mode,
        "steps": args.steps,
        "val_ce": round(full_val, 5),
        "val_ppl": round(math.exp(min(full_val, 20.0)), 3),
        "ms_per_step": round(med_ms, 2),
        "tok_per_s": round(med_toks, 1),
        "total_tokens": args.steps * toks_step,
    })
    print(f"=== DONE {run_id} | Full 10.5M Val CE = {full_val:.5f} | Median {med_ms:.1f} ms/st -> {log_p} ===", flush=True)


if __name__ == "__main__":
    main()
