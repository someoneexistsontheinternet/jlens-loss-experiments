"""
PyTorch / ROCm MI325X Implementation of the 8-Horizon Focusing-Wave Stein `J-Lens`:
  1. 10% Uncapped Hardest + 5% Easiest Token Selection (rounded up to 256-token tile blocks, multiple of 8 blocks).
  2. 8-Token Log-Gaussian Focusing Wave (`L02 -> L11` space-time bucket brigade at rho = 3 / sqrt(5 * pi) ~ 0.757):
       - Starts exact uniform `[0.125 x 8]` at `L02` (~10% depth).
       - Focuses and sweeps inward `k=4,5 -> k=3 -> k=2 -> k=1`, congregating onto `k=1` at `L12`.
  3. Zero-Param Closed-Form Multi-Horizon Stein Lenses `J_l^(1..8)` solved simultaneously from ONE `D x D` Cholesky factor per layer!
"""
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

RHO_HARMONIC = 3.0 / math.sqrt(5.0 * math.pi)  # 0.75694...
EOT_TOKEN    = 50256


def pure_rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)).to(x.dtype)


@torch.compile
def _fused_sampled_softcap_ce(
    h_l: torch.Tensor,
    w_y: torch.Tensor,
    w_cand: torch.Tensor,
    y_l: torch.Tensor,
    cand_ids: torch.Tensor,
    log_kc: torch.Tensor,
    eff_w_l: torch.Tensor,
    softcap: float,
) -> torch.Tensor:
    raw_true = (h_l.float() * w_y.float()).sum(dim=-1)
    raw_cand = torch.matmul(h_l, w_cand.t()).float()
    if softcap > 0.0:
        inv_sc = 1.0 / softcap
        lg_true = softcap * torch.tanh(raw_true * inv_sc)
        lg_cand = softcap * torch.tanh(raw_cand * inv_sc)
    else:
        lg_true, lg_cand = raw_true, raw_cand

    is_same  = (cand_ids.unsqueeze(0) == y_l.unsqueeze(1))
    adj_cand = torch.where(is_same, torch.full_like(lg_cand, -1e9), lg_cand - log_kc.unsqueeze(0))
    lse_all  = torch.logaddexp(lg_true, torch.logsumexp(adj_cand, dim=-1))
    nll_l    = lse_all - lg_true
    return (nll_l * eff_w_l).sum() / eff_w_l.sum().clamp_min(1.0)


class Wave8SteinJLensController:
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        vocab_pad: int,
        micro_tokens: int,
        device: torch.device,
        unigram_path: Path = None,
        start_layer: int = 1,
        horizons: int = 8,
        hard_frac: float = 0.10,
        easy_frac: float = 0.05,
        k_vocab: int = 4096,
        ridge: float = 0.10,
        ema: float = 0.90,
        softcap: float = 30.0,
    ):
        self.n_layers    = n_layers
        self.d_model     = d_model
        self.vocab_pad   = vocab_pad
        self.start_layer = start_layer   # 1 -> starts at L02 (~10% depth), supervising L02..L11 (10 layers)
        self.n_jlens     = n_layers - 1 - start_layer
        self.horizons    = horizons
        self.k_vocab     = k_vocab
        self.ridge       = ridge
        self.ema         = ema
        self.softcap     = softcap
        self.device      = device

        # 1. Build 256-aligned 8-Horizon Wave Schedule
        k_hard    = int(math.ceil((hard_frac * micro_tokens) / 256.0)) * 256
        k_easy    = int(math.ceil((easy_frac * micro_tokens) / 256.0)) * 256
        n_blk_raw = (k_hard + k_easy) // 256
        n_blk     = int(math.ceil(n_blk_raw / float(horizons))) * horizons
        k_easy   += (n_blk - n_blk_raw) * 256
        k_tot     = k_hard + k_easy

        self.k_hard = k_hard
        self.k_easy = k_easy
        self.k_tot  = k_tot
        self.n_blk  = n_blk

        M_s = n_layers - 1 - start_layer
        tok_h_idx = np.zeros((self.n_jlens, k_tot), dtype=np.int64)
        tok_imp_w = np.zeros((self.n_jlens, k_tot), dtype=np.float32)
        self.slices_per_layer = []

        for l_i in range(self.n_jlens):
            m = l_i
            u = m / float(M_s)
            beta_l = 0.0 if m == 0 else (2.2 * (u ** 0.65) + 3.2 * (u ** 2))
            mu_l   = 1.0 + 4.2 * ((RHO_HARMONIC ** (0.85 * m) - RHO_HARMONIC ** (0.85 * M_s)) / (1.0 - RHO_HARMONIC ** (0.85 * M_s)))
            logits = np.array([-beta_l * ((math.log(k) - math.log(mu_l)) ** 2) for k in range(1, horizons + 1)], dtype=np.float64)
            logits -= logits.max()
            w = np.exp(logits); w /= w.sum()

            raw_b = n_blk * w
            fl_b  = np.floor(raw_b).astype(np.int32)
            rem   = int(n_blk - fl_b.sum())
            if rem > 0:
                fl_b[np.argsort(-(raw_b - fl_b))[:rem]] += 1

            active_w_sum = float(w[fl_b > 0].sum())
            cursor = 0
            slices_l = []
            for h_k in range(horizons):
                nb = int(fl_b[h_k])
                if nb > 0:
                    nt = nb * 256
                    iw = float((w[h_k] / active_w_sum) / (nb / float(n_blk)))
                    tok_h_idx[l_i, cursor : cursor + nt] = h_k
                    tok_imp_w[l_i, cursor : cursor + nt] = iw
                    slices_l.append((h_k, cursor, cursor + nt))
                    cursor += nt
            self.slices_per_layer.append(slices_l)

        self.tok_h_idx = torch.from_numpy(tok_h_idx).to(device)
        self.tok_imp_w = torch.from_numpy(tok_imp_w).to(device)

        # 2. Initialize zero-param Stein lens buffer (N_JL, 8, D, D) to identity
        eye = torch.eye(d_model, dtype=torch.float32, device=device)
        self.jlens_mat = eye.view(1, 1, d_model, d_model).expand(self.n_jlens, horizons, d_model, d_model).clone()

        # 3. Unigram^0.75 proposal distribution for 4,096-way importance-sampled softmax
        if unigram_path is not None and Path(unigram_path).is_file():
            cnts = np.load(unigram_path)[:50257].astype(np.float64)
        else:
            cnts = np.ones((50257,), dtype=np.float64)
        p_v = np.maximum(cnts, 1.0) ** 0.75
        p_v /= p_v.sum()
        self.vocab_log_q = torch.from_numpy(np.log(np.maximum(p_v, 1e-12)).astype(np.float32)).to(device)

    @torch.no_grad()
    def sample_vocab_candidates(self) -> torch.Tensor:
        u = torch.empty_like(self.vocab_log_q).uniform_(1e-7, 1.0 - 1e-7)
        gumbel = -torch.log(-torch.log(u))
        _, cand_ids = torch.topk(self.vocab_log_q + gumbel, self.k_vocab)
        return cand_ids

    @torch.no_grad()
    def select_hard10_easy5(self, yb: torch.Tensor, tok_nll_det: torch.Tensor):
        B, T = yb.shape
        H    = self.horizons
        T_v  = T - (H - 1)

        # Valid positions t < T - 7 with non-EOT k=1 target
        t_grid = torch.arange(T, device=yb.device)[None, :].expand(B, T)
        valid  = (t_grid < T_v) & (yb != EOT_TOKEN)
        valid_1d = valid.reshape(-1)

        # Uncapped hardest 10% (invalid masked to -inf) + easiest 5% (invalid masked to +inf)
        nll_hard = torch.where(valid_1d, tok_nll_det, torch.full_like(tok_nll_det, -1e9))
        _, hard_idx = torch.topk(nll_hard, self.k_hard, largest=True)

        is_hard = torch.zeros_like(valid_1d)
        is_hard[hard_idx] = True
        nll_easy = torch.where(valid_1d & (~is_hard), tok_nll_det, torch.full_like(tok_nll_det, 1e9))
        _, easy_idx = torch.topk(nll_easy, self.k_easy, largest=False)

        sel_raw = torch.cat([hard_idx, easy_idx], dim=0)
        perm    = torch.randperm(self.k_tot, device=yb.device)
        sel_idx = sel_raw[perm]

        b_idx = sel_idx // T
        t_idx = sel_idx % T
        offs  = torch.arange(H, device=yb.device)[None, :]
        t_kh  = (t_idx[:, None] + offs).clamp_max(T - 1)
        y_kh  = yb[b_idx[:, None], t_kh]
        val_kh = torch.cumprod((y_kh != EOT_TOKEN).float(), dim=-1)
        return sel_idx, y_kh, val_kh

    @torch.no_grad()
    def refresh_lenses(self, interm_list: list[torch.Tensor], x_raw_final: torch.Tensor, yb: torch.Tensor):
        B, T = yb.shape
        H    = self.horizons
        T_v  = T - (H - 1)
        M_v  = B * T_v
        K_r  = min(4096, M_v)
        ridx = torch.randint(0, M_v, (K_r,), device=yb.device)

        x_s = torch.stack(
            [pure_rms_norm(x_l[:, :T_v, :].reshape(M_v, self.d_model)[ridx]).float()
             for x_l in interm_list[self.start_layer :]],
            dim=0,
        )  # (N_JL, K_r, D)
        xf_win = torch.stack([x_raw_final[:, h : T_v + h, :] for h in range(H)], dim=2).reshape(M_v, H, self.d_model)[ridx]
        vw_win = torch.cumprod(
            (torch.stack([yb[:, h : T_v + h] for h in range(H)], dim=-1) != EOT_TOKEN).float(), dim=-1
        ).reshape(M_v, H)[ridx]

        y_fh  = pure_rms_norm(xf_win).float() * vw_win.unsqueeze(-1)
        inv_n = 1.0 / float(K_r)
        eye   = torch.eye(self.d_model, dtype=torch.float32, device=yb.device).unsqueeze(0)
        cxx   = torch.einsum("lkd,lke->lde", x_s, x_s) * inv_n + self.ridge * eye
        cnt_h = vw_win.sum(dim=0).clamp_min(1.0).view(1, 1, H, 1)
        cxy   = (torch.einsum("lkd,khf->ldhf", x_s, y_fh) / cnt_h).reshape(self.n_jlens, self.d_model, H * self.d_model)

        chol   = torch.linalg.cholesky(cxx)
        w_flat = torch.cholesky_solve(cxy, chol)
        w_star = w_flat.view(self.n_jlens, self.d_model, H, self.d_model).permute(0, 2, 1, 3).contiguous()
        self.jlens_mat.mul_(self.ema).add_(w_star, alpha=(1.0 - self.ema))

    def compute_loss(
        self,
        interm_list: list[torch.Tensor],
        out_norm_weight: torch.Tensor,
        tok_emb_weight: torch.Tensor,
        yb: torch.Tensor,
        tok_nll_det: torch.Tensor,
        cand_ids: torch.Tensor,
    ):
        M = yb.numel()
        sel_idx, y_kh, val_kh = self.select_hard10_easy5(yb, tok_nll_det)

        x_all  = torch.stack([x_l.reshape(M, self.d_model)[sel_idx] for x_l in interm_list[self.start_layer - 1 :]], dim=0)
        x_prev = x_all[:-1].detach()
        x_stk  = pure_rms_norm(x_prev + (x_all[1:] - x_prev)).to(torch.bfloat16)

        j_det  = self.jlens_mat.detach().to(torch.bfloat16)
        out_sc = out_norm_weight.detach().to(torch.bfloat16)
        w_emb  = tok_emb_weight.detach().to(torch.bfloat16)
        w_cand = w_emb[cand_ids]
        log_kc = math.log(float(self.k_vocab)) + self.vocab_log_q[cand_ids]

        row_ar = torch.arange(self.k_tot, device=yb.device).unsqueeze(0)
        y_lk   = y_kh[row_ar, self.tok_h_idx]
        val_lk = val_kh[row_ar, self.tok_h_idx]
        eff_w  = self.tok_imp_w * val_lk

        layer_losses = []
        for l_i, slices_l in enumerate(self.slices_per_layer):
            parts = [torch.matmul(x_stk[l_i, c0:c1], j_det[l_i, h_k]) for (h_k, c0, c1) in slices_l]
            h_l   = (torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]) * out_sc.unsqueeze(0)
            y_l   = y_lk[l_i]
            layer_losses.append(
                _fused_sampled_softcap_ce(h_l, w_emb[y_l], w_cand, y_l, cand_ids, log_kc, eff_w[l_i], float(self.softcap))
            )

        j_loss = torch.stack(layer_losses).mean()
        return j_loss, j_loss.detach()
