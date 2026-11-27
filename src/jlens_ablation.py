"""
Causal-Ablation Workspace `J-Lens` (`jlens_mode = "ablation"`):
  1. Reads clean intermediate hidden states `h_l` through a frozen `J-Lens` snapshot (`J_l`, activation covariance `Sigma_l`,
     and frozen `[out_norm, tok_emb]` softcapped readout).
  2. Greedily selects up to `k` (`8` or `16`) diverse active token hypotheses from `top-64` (`|cos| <= max_cosine` in
     whitened/output space when `metric="cov"`).
  3. Suppresses each candidate direction via covariance-matched oblique projection `d_i = Sigma_l @ (J_l^T U_i)` (plus 1
     joint all-`k` suppression probe `U_all` for OR/AND coalition conservation), optionally restoring `||x||_2`.
  4. Evaluates signed causal usefulness `u_i = score(ablated_i) - score(clean)` under `no_grad` using an exact
     KV-cached `W`-token local causal lookahead window (`[p .. p + W - 1]`), cutting probe FLOPs >20x with 0 approximation:
       - At `anchor_layer` (e.g. `L09 -> L12`): `score` is downstream gold softcapped NLL over the `W`-token window.
       - At `early_layer`  (e.g. `L03 -> L09`): `score` stops at `anchor_layer` (`W=1` at `p`) and evaluates `anchor`'s
         hop-normalized fixed causal feature objective (`score_at_anchor`).
  5. Differentiates ONLY the clean `log_probs` path (`stopgrad` on all probe usefulnesses), supporting:
       - `superpos_unlikelihood` (default): 1-vs-background non-cannibalizing positive NLL + bounded unlikelihood on `u_i < 0`.
       - `unlikelihood`          : categorical positive NLL + bounded unlikelihood on `u_i < 0`.
       - `signed_raw`            : literal signed `-sum_i u_i * log q_i` control.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import triton
import triton.language as tl

from model_70m import apply_rope

RHO_HARMONIC = 3.0 / math.sqrt(5.0 * math.pi)  # 0.75694...
EOT_TOKEN    = 50256


@triton.jit
def _softcap_lse_nograd_kernel(
    logits_ptr, stride_m, lse_ptr,
    V: tl.constexpr, SOFTCAP: tl.constexpr, INV_SOFTCAP: tl.constexpr, BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * stride_m
    m_i = -float("inf")
    d_i = 0.0
    LOG2E: tl.constexpr = 1.4426950408889634
    for start_v in range(0, V, BLOCK_V):
        offs_v = start_v + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V
        raw = tl.load(row_ptr + offs_v, mask=mask_v, other=-float("inf")).to(tl.float32)
        if SOFTCAP > 0.0:
            z = tl.minimum(tl.maximum(raw * INV_SOFTCAP, -20.0), 20.0)
            e2z = tl.math.exp2(2.0 * LOG2E * z)
            lg = tl.where(mask_v, SOFTCAP * ((e2z - 1.0) / (e2z + 1.0)), -float("inf"))
        else:
            lg = raw
        blk_max = tl.max(lg, axis=0)
        m_new = tl.maximum(m_i, blk_max)
        d_i = d_i * tl.math.exp2((m_i - m_new) * LOG2E) + tl.sum(tl.math.exp2((lg - m_new) * LOG2E), axis=0)
        m_i = m_new
    tl.store(lse_ptr + row, m_i + tl.math.log(d_i))


@triton.jit
def _greedy_diverse_accept_kernel(
    sim_norms_ptr, cos_mat_ptr, accepted_ptr,
    M_pool, k_max, max_cos, BLOCK_M: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_M)
    acc_mask = tl.zeros((BLOCK_M,), dtype=tl.int32)
    n_acc = 0
    for j in range(0, M_pool):
        sn = tl.load(sim_norms_ptr + b * M_pool + j)
        cos_row = tl.load(cos_mat_ptr + b * M_pool * M_pool + j * M_pool + offs, mask=(offs < j), other=0.0)
        conflict = tl.max(tl.where((offs < j) & (acc_mask > 0) & (cos_row > max_cos), 1, 0), axis=0)
        take = (sn > 0.5) & (conflict == 0) & (n_acc < k_max)
        take_i = tl.where(take, 1, 0)
        acc_mask = tl.where(offs == j, take_i, acc_mask)
        n_acc += take_i
    tl.store(accepted_ptr + b * M_pool + offs, acc_mask.to(tl.int8), mask=(offs < M_pool))


def pure_rms_norm_fp32(x: Tensor, eps: float = 1e-6) -> Tensor:
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)


class FrozenSoftcapReadout(nn.Module):
    """Frozen snapshot of final RMSNorm + tied unembedding + logit softcap (with fast BF16-MFMA + Triton no-grad path)."""

    def __init__(self, norm_weight: Tensor, unemb_weight: Tensor, softcap: float = 30.0, eps: float = 1e-6):
        super().__init__()
        self.softcap = float(softcap)
        self.eps     = float(eps)
        self.register_buffer("norm_w", norm_weight.detach().float().clone())
        self.register_buffer("U", unemb_weight.detach().float().clone())
        self.register_buffer("U_bf16", unemb_weight.detach().to(torch.bfloat16).clone())

    @torch.no_grad()
    def copy_from_(self, norm_weight: Tensor, unemb_weight: Tensor):
        self.norm_w.copy_(norm_weight.detach().float())
        self.U.copy_(unemb_weight.detach().float())
        self.U_bf16.copy_(unemb_weight.detach().to(torch.bfloat16))

    def logits(self, h: Tensor) -> Tensor:
        with torch.amp.autocast(device_type=h.device.type, enabled=False):
            hf = h.float()
            hn = hf * torch.rsqrt(hf.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.norm_w
            raw = F.linear(hn, self.U)
            if self.softcap > 0.0:
                return self.softcap * torch.tanh(raw * (1.0 / self.softcap))
            return raw

    def log_probs(self, h: Tensor) -> Tensor:
        return self.logits(h).log_softmax(dim=-1)

    @torch.no_grad()
    def fast_nograd_target_nll(self, h_2d: Tensor, targets_1d: Tensor, chunk_sz: int = 8192) -> Tensor:
        """Fast BF16-MFMA + 1-pass Triton online-LSE gold NLL for `[N, D]` probe states."""
        if not h_2d.is_cuda or self.U_bf16.shape[0] < 1024:
            return F.cross_entropy(self.logits(h_2d), targets_1d, reduction="none")
        N, _ = h_2d.shape
        V = self.U_bf16.shape[0]
        inv_sc = (1.0 / self.softcap) if self.softcap > 0.0 else 0.0
        out_nll = torch.empty((N,), dtype=torch.float32, device=h_2d.device)
        buf = torch.empty((min(N, chunk_sz), V), dtype=torch.bfloat16, device=h_2d.device)
        lse = torch.empty((min(N, chunk_sz),), dtype=torch.float32, device=h_2d.device)
        for s in range(0, N, chunk_sz):
            e = min(s + chunk_sz, N)
            nr = e - s
            hf = h_2d[s:e].float()
            hn = (hf * torch.rsqrt(hf.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.norm_w).to(torch.bfloat16)
            b_sl = buf[:nr]
            l_sl = lse[:nr]
            torch.matmul(hn, self.U_bf16.t(), out=b_sl)
            _softcap_lse_nograd_kernel[(nr,)](
                b_sl, b_sl.stride(0), l_sl, V=V, SOFTCAP=self.softcap, INV_SOFTCAP=inv_sc, BLOCK_V=4096, num_warps=8
            )
            raw_t = b_sl.gather(1, targets_1d[s:e].unsqueeze(1)).squeeze(1).float()
            lg_t = (self.softcap * torch.tanh(raw_t * inv_sc)) if self.softcap > 0.0 else raw_t
            out_nll[s:e] = l_sl - lg_t
        return out_nll

    @torch.no_grad()
    def fast_nograd_selected_logprobs(self, h_2d: Tensor, sel_ids_2d: Tensor, chunk_sz: int = 8192) -> Tensor:
        """Returns exact `log_probs(h_2d).gather(-1, sel_ids_2d)` `[N, k]` without materializing `[N, V]` FP32 log_softmax!"""
        if not h_2d.is_cuda or self.U_bf16.shape[0] < 1024:
            return self.log_probs(h_2d).gather(-1, sel_ids_2d)
        N, _ = h_2d.shape
        k = sel_ids_2d.shape[1]
        V = self.U_bf16.shape[0]
        inv_sc = (1.0 / self.softcap) if self.softcap > 0.0 else 0.0
        out_lq = torch.empty((N, k), dtype=torch.float32, device=h_2d.device)
        buf = torch.empty((min(N, chunk_sz), V), dtype=torch.bfloat16, device=h_2d.device)
        lse = torch.empty((min(N, chunk_sz),), dtype=torch.float32, device=h_2d.device)
        for s in range(0, N, chunk_sz):
            e = min(s + chunk_sz, N)
            nr = e - s
            hf = h_2d[s:e].float()
            hn = (hf * torch.rsqrt(hf.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.norm_w).to(torch.bfloat16)
            b_sl = buf[:nr]
            l_sl = lse[:nr]
            torch.matmul(hn, self.U_bf16.t(), out=b_sl)
            _softcap_lse_nograd_kernel[(nr,)](
                b_sl, b_sl.stride(0), l_sl, V=V, SOFTCAP=self.softcap, INV_SOFTCAP=inv_sc, BLOCK_V=4096, num_warps=8
            )
            raw_k = b_sl.gather(1, sel_ids_2d[s:e]).float()
            lg_k = (self.softcap * torch.tanh(raw_k * inv_sc)) if self.softcap > 0.0 else raw_k
            out_lq[s:e] = lg_k - l_sl.unsqueeze(1)
        return out_lq


class WorkspaceJLens(nn.Module):
    """Layer lens `J_l` (`h_final ~ x @ J.T`) + running activation covariance `Sigma_l` + shared frozen readout."""

    def __init__(self, d_model: int, readout: FrozenSoftcapReadout, device: torch.device):
        super().__init__()
        self.d_model = d_model
        self.readout = readout
        self.register_buffer("J", torch.eye(d_model, dtype=torch.float32, device=device))
        self.register_buffer("Sigma", torch.eye(d_model, dtype=torch.float32, device=device))

    def log_probs(self, x: Tensor) -> Tensor:
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            return self.readout.log_probs(x.float() @ self.J.t())

    @torch.no_grad()
    def candidate_vectors(self, ids: Tensor, metric: str = "cov") -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
          w_read  : [B, M, D] dual readout covectors `w_i = U_i @ J`
          d_write : [B, M, D] primal intervention vectors (`w_i @ Sigma` if `metric=="cov"` else `w_i`)
          u_sim   : [B, M, D] unit vectors used for cosine-diversity filtering
        """
        with torch.amp.autocast(device_type=ids.device.type, enabled=False):
            u_tok  = self.readout.U[ids]              # [B, M, D]
            w_read = torch.matmul(u_tok, self.J)      # [B, M, D]
            if metric == "cov":
                d_write = torch.matmul(w_read, self.Sigma)  # Sigma is symmetric
                # Under Mahalanobis metric Sigma, <w_i, Sigma w_j> induces inner product; filter in output/whitened space:
                denom = (w_read * d_write).sum(dim=-1, keepdim=True).clamp_min(1e-16).sqrt()
                # Use output-space normalized `u_tok` when well-conditioned, or Sigma-whitened `w_read`:
                u_norm = u_tok.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                u_sim  = u_tok / u_norm
                # Guard degenerate zero-sensitivity rows:
                u_sim  = torch.where(denom > 1e-8, u_sim, torch.zeros_like(u_sim))
            else:
                d_write = w_read
                w_norm  = w_read.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                u_sim   = torch.where(w_read.norm(dim=-1, keepdim=True) > 1e-8, w_read / w_norm, torch.zeros_like(w_read))
            return w_read, d_write, u_sim


@torch.no_grad()
def select_diverse_fast(
    logq: Tensor,
    lens: WorkspaceJLens,
    pool_size: int = 64,
    k: int = 8,
    max_cosine: float = 0.85,
    metric: str = "cov",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    GPU-vectorized greedy cosine-diversity filter (bit-equivalent to sequential reference loop when metric='euclidean').
    Returns:
      ids     : [B, k]
      w_read  : [B, k, D]
      d_write : [B, k, D]
      valid   : [B, k] bool
    """
    M_pool = min(pool_size, logq.shape[-1])
    with torch.amp.autocast(device_type=logq.device.type, enabled=False):
        pool   = logq.float().topk(M_pool, dim=-1).indices          # [B, M]
        w_pool, d_pool, u_sim = lens.candidate_vectors(pool, metric=metric)
        B, _, D = w_pool.shape

        sim_norms = u_sim.float().norm(dim=-1)                      # [B, M]
        cos_mat   = torch.matmul(u_sim.float(), u_sim.float().transpose(-1, -2)).abs()  # [B, M, M]

    if logq.is_cuda and M_pool <= 64:
        acc_i8 = torch.empty((B, M_pool), dtype=torch.int8, device=logq.device)
        _greedy_diverse_accept_kernel[(B,)](
            sim_norms.contiguous(), cos_mat.contiguous(), acc_i8,
            M_pool, k, float(max_cosine), BLOCK_M=64, num_warps=2,
        )
        accepted = acc_i8.bool()
    else:
        accepted = torch.zeros((B, M_pool), dtype=torch.bool, device=logq.device)
        n_acc    = torch.zeros((B,), dtype=torch.long, device=logq.device)
        for j in range(M_pool):
            ok_len = sim_norms[:, j] > 0.5
            if j == 0:
                conflict = torch.zeros((B,), dtype=torch.bool, device=logq.device)
            else:
                prev_acc = accepted[:, :j]
                c_prev   = cos_mat[:, j, :j]
                conflict = (c_prev > max_cosine) & prev_acc
                conflict = conflict.any(dim=-1)
            take = ok_len & (~conflict) & (n_acc < k)
            accepted[:, j] = take
            n_acc = n_acc + take.long()

    # Pack up to `k` accepted entries per batch row in original score order:
    slot_idx = torch.cumsum(accepted.long(), dim=-1) - 1        # [B, M]
    ids_out  = torch.zeros((B, k), dtype=torch.long, device=logq.device)
    w_out    = torch.zeros((B, k, D), dtype=torch.float32, device=logq.device)
    d_out    = torch.zeros((B, k, D), dtype=torch.float32, device=logq.device)
    val_out  = torch.zeros((B, k), dtype=torch.bool, device=logq.device)

    b_idx, m_idx = torch.nonzero(accepted, as_tuple=True)
    if b_idx.numel() > 0:
        s_idx = slot_idx[b_idx, m_idx]
        ids_out[b_idx, s_idx] = pool[b_idx, m_idx]
        w_out[b_idx, s_idx]   = w_pool[b_idx, m_idx]
        d_out[b_idx, s_idx]   = d_pool[b_idx, m_idx]
        val_out[b_idx, s_idx] = True

    return ids_out, w_out, d_out, val_out


@torch.no_grad()
def suppress_oblique_single_and_joint(
    x: Tensor,
    w_read: Tensor,
    d_write: Tensor,
    valid: Tensor,
    strength: float = 1.0,
    match_norm: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        B, k, D = w_read.shape
        xf = x.float()
        w_read = w_read.float()
        d_write = d_write.float()
        old_norm = xf.norm(dim=-1, keepdim=True)

        if strength <= 0.0:
            clean_rep = x.unsqueeze(1).expand(B, k + 2, D).contiguous()
            return clean_rep, valid.clone(), valid.any(dim=-1)

        wx    = torch.einsum("bkd,bd->bk", w_read, xf)
        wd    = (w_read * d_write).sum(dim=-1)
        coeff = torch.where(valid & (wd > 1e-12), strength * (wx / wd.clamp_min(1e-12)), torch.zeros_like(wx))
        ed_k  = xf.unsqueeze(1) - coeff.unsqueeze(-1) * d_write
        usable_k = valid & torch.isfinite(ed_k).all(dim=-1) & (wd > 1e-12)

        if match_norm:
            new_n = ed_k.norm(dim=-1, keepdim=True)
            old_n = old_norm.unsqueeze(1)
            ok_n  = (old_n[..., 0] > 1e-8) & (new_n[..., 0] > 1e-6 * old_n[..., 0].clamp_min(1e-8))
            usable_k = usable_k & ok_n
            ed_k  = ed_k * (old_n / new_n.clamp_min(1e-8))
        ed_k = torch.where(usable_k.unsqueeze(-1), ed_k, xf.unsqueeze(1))

        w_m = torch.where(usable_k.unsqueeze(-1), w_read, torch.zeros_like(w_read))
        d_m = torch.where(usable_k.unsqueeze(-1), d_write, torch.zeros_like(d_write))
        gram  = torch.matmul(w_m, d_m.transpose(-1, -2))
        diag_scale = gram.diagonal(dim1=-2, dim2=-1).abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
        eye_k = torch.eye(k, dtype=torch.float32, device=x.device).unsqueeze(0)
        gram_reg = gram + (1e-4 * diag_scale.unsqueeze(-1)) * eye_k
        rhs   = torch.einsum("bkd,bd->bk", w_m, xf).unsqueeze(-1)
        alpha = torch.linalg.solve(gram_reg, rhs).squeeze(-1)
        alpha = torch.where(usable_k, alpha, torch.zeros_like(alpha))
        ed_all = xf - strength * torch.einsum("bk,bkd->bd", alpha, d_m)
        usable_all = usable_k.any(dim=-1) & torch.isfinite(ed_all).all(dim=-1)

        if match_norm:
            new_na = ed_all.norm(dim=-1, keepdim=True)
            ok_na  = (old_norm[:, 0] > 1e-8) & (new_na[:, 0] > 1e-6 * old_norm[:, 0].clamp_min(1e-8))
            usable_all = usable_all & ok_na
            ed_all = ed_all * (old_norm / new_na.clamp_min(1e-8))
        ed_all = torch.where(usable_all.unsqueeze(-1), ed_all, xf)

        probes = torch.cat([xf.unsqueeze(1), ed_k, ed_all.unsqueeze(1)], dim=1).to(x.dtype)
        return probes, usable_k, usable_all


def compute_feature_loss_pregathered(
    lq_sel: Tensor,
    weights: Tensor,
    valid: Tensor,
    loss_kind: str = "superpos_unlikelihood",
) -> Tensor:
    w = torch.where(valid, weights.detach().float(), torch.zeros_like(weights, dtype=torch.float32))
    lq_f = lq_sel.float()

    if loss_kind == "signed_raw":
        return -(w * lq_f).sum(dim=-1)

    w_pos = F.relu(w)
    w_neg = F.relu(-w)
    q_sel = lq_f.exp().clamp(0.0, 1.0 - 1e-7)
    neg_term = -(w_neg * torch.log1p(-q_sel)).sum(dim=-1)

    if loss_kind == "unlikelihood":
        pos_term = -(w_pos * lq_f).sum(dim=-1)
        return pos_term + neg_term

    pos_mask = (w > 0.0) & valid
    sum_q_pos = (q_sel * pos_mask.float()).sum(dim=-1, keepdim=True)
    q_bg = (1.0 - sum_q_pos).clamp_min(1e-6)
    log_q_bg  = torch.log(q_bg)
    log_p_ind = lq_f - torch.logaddexp(lq_f, log_q_bg.expand_as(lq_f))
    pos_term  = -(w_pos * log_p_ind).sum(dim=-1)
    return pos_term + neg_term


def compute_feature_loss(
    logq: Tensor,
    ids: Tensor,
    weights: Tensor,
    valid: Tensor,
    loss_kind: str = "superpos_unlikelihood",
) -> Tensor:
    lq_sel = logq.float().gather(-1, ids)
    return compute_feature_loss_pregathered(lq_sel, weights, valid, loss_kind=loss_kind)


@torch.no_grad()
def extract_clean_kv_for_blocks(
    blocks: nn.ModuleList,
    clean_inputs: list[Tensor],
    b_indices: Tensor,
    cos_full: Tensor,
    sin_full: Tensor,
) -> list[tuple[Tensor, Tensor]]:
    """Computes clean `(K, V)` caches `[B_p, H, T, Dh]` directly from stored block inputs `clean_inputs`."""
    kv_list = []
    T = clean_inputs[0].shape[1]
    cos_t, sin_t = cos_full[:, :, :T, :], sin_full[:, :, :T, :]
    for blk, x_in in zip(blocks, clean_inputs):
        xb = x_in[b_indices]                                    # [B_p, T, D]
        B_p, _, D = xb.shape
        H, Dh = blk.cfg.n_heads, blk.cfg.head_dim
        qkv = blk.wqkv(blk.attn_norm(xb)).view(B_p, T, 3, H, Dh)
        k_c = apply_rope(blk.k_norm(qkv[:, :, 1]).transpose(1, 2), cos_t, sin_t)
        v_c = qkv[:, :, 2].transpose(1, 2)
        kv_list.append((k_c, v_c))
    return kv_list


@torch.no_grad()
def advance_local_window_kv(
    blocks: nn.ModuleList,
    kv_list: list[tuple[Tensor, Tensor]],
    h_win: Tensor,
    positions: Tensor,
    cos_full: Tensor,
    sin_full: Tensor,
) -> Tensor:
    """
    Exact KV-cached causal window propagation!
    Advances `P` parallel probe trajectories of width `W` (`h_win`: `[B_p, P, W, D]`) starting at per-example
    token index `positions` (`[B_p]`), attending to shared clean prefix `0 .. positions[b]-1` + probe-local causal `[0..W-1]`.
    Mathematically identical to cloning `[B_p, T, D]` `P` times and running full-sequence `blocks`, at ~250x fewer FLOPs!
    """
    B_p, P, W, D = h_win.shape
    T = kv_list[0][0].shape[2]
    H, Dh = blocks[0].cfg.n_heads, blocks[0].cfg.head_dim
    inv_sqrt_dh = 1.0 / math.sqrt(float(Dh))

    w_offs = torch.arange(W, device=h_win.device)
    win_pos = positions[:, None] + w_offs[None, :]              # [B_p, W]
    cos_Sq  = cos_full[0, 0][win_pos][:, None, None, :, :]      # [B_p, 1, 1, W, Dh//2]
    sin_Sq  = sin_full[0, 0][win_pos][:, None, None, :, :]      # [B_p, 1, 1, W, Dh//2]

    t_ar = torch.arange(T, device=h_win.device)[None, :]        # [1, T]
    pre_valid = (t_ar < positions[:, None])                     # [B_p, T]
    loc_causal = torch.ones((W, W), dtype=torch.bool, device=h_win.device).tril(0)  # [W, W]

    x = h_win
    for blk, (k_clean, v_clean) in zip(blocks, kv_list):
        x_flat = x.reshape(B_p * P, W, D)
        qkv = blk.wqkv(blk.attn_norm(x_flat)).view(B_p, P, W, 3, H, Dh)
        q_w = apply_rope(blk.q_norm(qkv[:, :, :, 0]).transpose(2, 3), cos_Sq, sin_Sq)  # [B_p, P, H, W, Dh]
        k_w = apply_rope(blk.k_norm(qkv[:, :, :, 1]).transpose(2, 3), cos_Sq, sin_Sq)  # [B_p, P, H, W, Dh]
        v_w = qkv[:, :, :, 2].transpose(2, 3)                                          # [B_p, P, H, W, Dh]

        # Shared prefix Q @ K_clean^T without replicating K_clean across P (BF16 MFMA -> FP32 softmax):
        q_pw = q_w.permute(0, 2, 1, 3, 4).reshape(B_p, H, P * W, Dh)
        sc_pre = torch.matmul(q_pw.to(x.dtype), k_clean.to(x.dtype).transpose(-1, -2)).float() * inv_sqrt_dh
        sc_pre = sc_pre.masked_fill(~pre_valid[:, None, None, :], -float("inf"))
        sc_pre = sc_pre.view(B_p, H, P, W, T).permute(0, 2, 1, 3, 4)                   # [B_p, P, H, W, T]

        sc_loc = torch.matmul(q_w.to(x.dtype), k_w.to(x.dtype).transpose(-1, -2)).float() * inv_sqrt_dh
        sc_loc = sc_loc.masked_fill(~loc_causal[None, None, None, :, :], -float("inf"))

        probs_all = torch.softmax(torch.cat([sc_pre, sc_loc], dim=-1), dim=-1).to(x.dtype)
        pr_pre = probs_all[..., :T].permute(0, 2, 1, 3, 4).reshape(B_p, H, P * W, T)
        pr_loc = probs_all[..., T:]

        out_pre = torch.matmul(pr_pre, v_clean.to(x.dtype)).view(B_p, H, P, W, Dh).permute(0, 2, 1, 3, 4)
        out_loc = torch.matmul(pr_loc, v_w.to(x.dtype))
        y_attn  = (out_pre + out_loc).transpose(2, 3).reshape(B_p * P, W, D)

        x_flat = x_flat + blk.wo(y_attn)
        g, u   = blk.wgu(blk.mlp_norm(x_flat)).chunk(2, dim=-1)
        x_flat = x_flat + blk.wdown(F.silu(g) * u)
        x = x_flat.view(B_p, P, W, D)

    return x


@torch.no_grad()
def calibrate_usefulness(
    raw_u: Tensor,
    u_all: Tensor,
    valid: Tensor,
    usable_all: Tensor,
    coalition_norm: bool = True,
) -> Tensor:
    """
    Applies `k + 1` coalition conservation (`OR` backup recovery & `AND` double-count damping):
      Rescales `|u_j|` softly toward `|U_all|` with factor clamped in `[0.25, 4.0]`.
    """
    u = torch.where(valid, raw_u, torch.zeros_like(raw_u))
    if not coalition_norm:
        return u
    s_ind  = u.abs().sum(dim=-1, keepdim=True)                  # [B, 1]
    target = torch.maximum(u_all.abs().unsqueeze(-1), 0.25 * s_ind)
    gamma  = (target / s_ind.clamp_min(1e-6)).clamp(0.25, 4.0)
    gamma  = torch.where((s_ind > 1e-6) & usable_all.unsqueeze(-1), gamma, torch.ones_like(gamma))
    return u * gamma


class CausalAblationJLensController:
    def __init__(
        self,
        raw_model: nn.Module,
        device: torch.device,
        early_layer: int = 3,
        anchor_layer: int = 9,
        probe_seqs: int = 64,
        pool_size: int = 64,
        k_dirs: int = 8,
        window: int = 4,
        max_cosine: float = 0.85,
        strength: float = 1.0,
        match_norm: bool = True,
        metric: str = "cov",
        loss_kind: str = "superpos_unlikelihood",
        coalition_norm: bool = True,
        hop_norm: bool = True,
        enable_early: bool = True,
        pos_mode: str = "hard_easy",
        detach_prev: bool = False,
        ridge: float = 0.10,
        ema: float = 0.90,
    ):
        assert 1 <= early_layer < anchor_layer < raw_model.cfg.n_layers
        self.cfg            = raw_model.cfg
        self.device         = device
        self.early_layer    = early_layer
        self.anchor_layer   = anchor_layer
        self.probe_seqs     = probe_seqs
        self.pool_size      = pool_size
        self.k_dirs         = k_dirs
        self.window         = window
        self.max_cosine     = max_cosine
        self.strength       = strength
        self.match_norm     = match_norm
        self.metric         = metric
        self.loss_kind      = loss_kind
        self.coalition_norm = coalition_norm
        self.hop_norm       = hop_norm
        self.enable_early   = enable_early
        self.pos_mode       = pos_mode
        self.detach_prev    = detach_prev
        self.ridge          = ridge
        self.ema            = ema
        self._initialized   = False

        self.readout = FrozenSoftcapReadout(
            raw_model.out_norm.weight, raw_model.tok_emb.weight, softcap=self.cfg.softcap
        ).to(device)
        self.lens_early  = WorkspaceJLens(self.cfg.d_model, self.readout, device).to(device)
        self.lens_anchor = WorkspaceJLens(self.cfg.d_model, self.readout, device).to(device)

        # Normalized geometric lookahead weights over `[p .. p + W - 1]`:
        w_vec = torch.tensor([RHO_HARMONIC ** m for m in range(window)], dtype=torch.float32, device=device)
        self.win_weights = w_vec / w_vec.sum()

    def estimate_call_flops(self, micro_batch: int) -> float:
        """Exact forward FLOPs per `compute_loss` call (clean KV extraction +KV-window probes + lens readouts)."""
        B_p = min(self.probe_seqs, micro_batch)
        T   = self.cfg.seq_len
        D   = self.cfg.d_model
        F_d = self.cfg.d_ff
        V   = self.cfg.vocab_pad
        P   = self.k_dirs + 2
        W_a = self.window
        n_anc_blk = self.cfg.n_layers - self.anchor_layer
        n_ear_blk = (self.anchor_layer - self.early_layer) if self.enable_early else 0

        # 1) Clean KV extraction (`wqkv`: 2 * B_p * T * D * 3D per block)
        kv_flops = (n_anc_blk + n_ear_blk) * (2.0 * B_p * T * D * (3 * D))
        # 2) Probe block linear + attention FLOPs (`2N` fwd per probe token + prefix/local QK^T & PV)
        blk_lin_2n = 2.0 * (D * 3 * D + D * D + D * 2 * F_d + F_d * D)
        anc_tok = B_p * P * W_a
        anc_flops = n_anc_blk * (anc_tok * blk_lin_2n + 4.0 * anc_tok * T * D)
        ear_tok = B_p * P * 1
        ear_flops = n_ear_blk * (ear_tok * blk_lin_2n + 4.0 * ear_tok * T * D)
        # 3) Readout logits (`fin_win` [B_p*P*W_a, V] + `at_anc` [B_p*P, V] + clean fwd+bwd `logq` [B_p, V])
        ro_anc = 2.0 * (anc_tok * D * V) + 6.0 * (B_p * (D * D + D * V))
        ro_ear = (2.0 * (ear_tok * D * V) + 6.0 * (B_p * (D * D + D * V))) if self.enable_early else 0.0
        return kv_flops + anc_flops + ear_flops + ro_anc + ro_ear

    @torch.no_grad()
    def refresh_snapshot(self, raw_model: nn.Module, interm_det: list[Tensor], x_raw_final: Tensor):
        """Updates frozen `[out_norm, tok_emb]` readout + Stein `J_l` (`x @ J.T ~ x_final`) + covariance `Sigma_l`."""
        with torch.amp.autocast(device_type=x_raw_final.device.type, enabled=False):
            self.readout.copy_from_(raw_model.out_norm.weight, raw_model.tok_emb.weight)

            B, T, D = x_raw_final.shape
            M_tot   = B * T
            K_r     = min(4096, M_tot)
            ridx    = torch.randint(0, M_tot, (K_r,), device=x_raw_final.device)
            y_fin   = x_raw_final.reshape(M_tot, D)[ridx].float()
            y_c     = y_fin - y_fin.mean(dim=0, keepdim=True)

            alpha = self.ema if self._initialized else 0.0
            eye   = torch.eye(D, dtype=torch.float32, device=x_raw_final.device)

            for layer_idx, lens in ((self.early_layer, self.lens_early), (self.anchor_layer, self.lens_anchor)):
                x_l = interm_det[layer_idx - 1].reshape(M_tot, D)[ridx].float()
                x_c = x_l - x_l.mean(dim=0, keepdim=True)
                cov = (x_c.t() @ x_c) * (1.0 / float(K_r))
                trace_avg = cov.diagonal().mean().clamp_min(1e-6)
                cov_reg   = cov + (self.ridge * trace_avg) * eye

                cxy   = (x_c.t() @ y_c) * (1.0 / float(K_r))
                chol  = torch.linalg.cholesky(cov_reg)
                w_map = torch.cholesky_solve(cxy, chol)
                j_new = w_map.t().contiguous()

                sig_new = cov / trace_avg + 0.02 * eye
                lens.J.mul_(alpha).add_(j_new, alpha=(1.0 - alpha))
                lens.Sigma.mul_(alpha).add_(sig_new, alpha=(1.0 - alpha))

            self._initialized = True

    @torch.no_grad()
    def select_probe_sequences_and_positions(self, yb: Tensor, tok_nll_det: Tensor) -> tuple[Tensor, Tensor]:
        """Picks `B_p = min(probe_seqs, B)` sequences and `1` high-signal position `p_b in [0, T - W]` per sequence."""
        B, T = yb.shape
        W    = self.window
        T_v  = T - W + 1
        B_p  = min(self.probe_seqs, B)
        nll_bt = tok_nll_det.view(B, T)[:, :T_v]

        # Valid starts: no EOT inside `[p .. p + W - 1]`
        valid_pos = torch.ones((B, T_v), dtype=torch.bool, device=yb.device)
        for m in range(W):
            valid_pos &= (yb[:, m : T_v + m] != EOT_TOKEN)

        if self.pos_mode == "uniform":
            b_sel = torch.randperm(B, device=yb.device)[:B_p]
            rnd   = torch.rand((B_p, T_v), device=yb.device).masked_fill(~valid_pos[b_sel], -1.0)
            pos   = rnd.argmax(dim=-1)
            return b_sel, pos

        # Score each position by lookahead max NLL over `[p .. p + W - 1]` (catches pre-hard setup tokens!)
        win_nlls = torch.stack([tok_nll_det.view(B, T)[:, m : T_v + m] for m in range(W)], dim=-1)
        look_max = win_nlls.amax(dim=-1).masked_fill(~valid_pos, -1e9)  # [B, T_v]

        seq_peak, best_hard_pos = look_max.max(dim=-1)                  # [B]
        n_hard = (2 * B_p) // 3
        n_easy = B_p - n_hard

        _, hard_seqs = torch.topk(seq_peak, n_hard, largest=True)
        rem_mask = torch.ones((B,), dtype=torch.bool, device=yb.device)
        rem_mask[hard_seqs] = False
        rem_seqs = torch.nonzero(rem_mask, as_tuple=True)[0]
        easy_seqs = rem_seqs[torch.randperm(rem_seqs.numel(), device=yb.device)[:n_easy]]

        # On easy sequences, pick lowest-NLL valid token (`> 0.05` nats so non-trivial bridge token)
        easy_scores = torch.where(
            valid_pos[easy_seqs] & (nll_bt[easy_seqs] > 0.05),
            nll_bt[easy_seqs],
            torch.full_like(nll_bt[easy_seqs], 1e9),
        )
        best_easy_pos = easy_scores.argmin(dim=-1)

        b_sel = torch.cat([hard_seqs, easy_seqs], dim=0)
        pos   = torch.cat([best_hard_pos[hard_seqs], best_easy_pos], dim=0)
        return b_sel, pos

    @torch._dynamo.disable
    def compute_loss(
        self,
        raw_model: nn.Module,
        interm: list[Tensor],
        yb: Tensor,
        tok_nll_det: Tensor,
    ) -> tuple[Tensor, Tensor, dict]:
        """
        Computes combined Anchor (`L09 -> L12`) + Early (`L03 -> L09`) causal-ablation auxiliary loss.
        Only the clean `interm` states receive gradients!
        """
        W = self.window
        k = self.k_dirs
        b_sel, pos = self.select_probe_sequences_and_positions(yb, tok_nll_det)
        B_p = b_sel.shape[0]

        # Clean live states at `anchor_layer` and `early_layer` (`[B_p, D]`)
        h_anc_seq = interm[self.anchor_layer - 1]
        if self.detach_prev and self.anchor_layer >= 2:
            prev_a = interm[self.anchor_layer - 2].detach()
            h_anc_seq = prev_a + (h_anc_seq - prev_a)
        x_anc_live = h_anc_seq[b_sel, pos]                      # [B_p, D], grad-attached!

        logq_anc = self.lens_anchor.log_probs(x_anc_live)       # [B_p, V], grad-attached!
        ids_a, w_a, d_a, val_a = select_diverse_fast(
            logq_anc.detach(), self.lens_anchor,
            pool_size=self.pool_size, k=k, max_cosine=self.max_cosine, metric=self.metric,
        )

        # --- 1. Anchor Causal Probes (`anchor_layer .. n_layers`, window `W`) ---
        with torch.no_grad():
            probes_a, usable_a, usable_all_a = suppress_oblique_single_and_joint(
                x_anc_live.detach(), w_a, d_a, val_a, strength=self.strength, match_norm=self.match_norm
            )  # [B_p, k+2, D]
            P = k + 2
            w_offs = torch.arange(W, device=yb.device)
            win_idx = pos[:, None] + w_offs[None, :]            # [B_p, W]
            h_anc_clean_win = interm[self.anchor_layer - 1].detach()[b_sel[:, None], win_idx]  # [B_p, W, D]
            h_win_a = h_anc_clean_win.unsqueeze(1).expand(B_p, P, W, self.cfg.d_model).clone()
            h_win_a[:, :, 0, :] = probes_a

            anc_blocks = raw_model.blocks[self.anchor_layer :]
            anc_inputs = [interm[m - 1].detach() for m in range(self.anchor_layer, self.cfg.n_layers)]
            kv_anc = extract_clean_kv_for_blocks(anc_blocks, anc_inputs, b_sel, raw_model.cos, raw_model.sin)
            fin_win = advance_local_window_kv(anc_blocks, kv_anc, h_win_a, pos, raw_model.cos, raw_model.sin)  # [B_p, P, W, D]

            # Gold softcapped NLL across `[p .. p + W - 1]` via BF16-MFMA + Triton online LSE:
            y_win = yb[b_sel[:, None], win_idx]                 # [B_p, W]
            flat_fin = fin_win.reshape(-1, self.cfg.d_model)
            flat_y   = y_win[:, None, :].expand(B_p, P, W).reshape(-1)
            nll_pw   = self.readout.fast_nograd_target_nll(flat_fin, flat_y).view(B_p, P, W)
            valid_y = (y_win != EOT_TOKEN).float()
            wt_y = valid_y * self.win_weights.unsqueeze(0)
            wt_y = wt_y / wt_y.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            scores_a = (nll_pw * wt_y.unsqueeze(1)).sum(dim=-1)  # [B_p, P]

            clean_sc_a = scores_a[:, 0:1]
            raw_u_a    = scores_a[:, 1 : k + 1] - clean_sc_a    # [B_p, k]
            u_all_a    = scores_a[:, k + 1] - scores_a[:, 0]    # [B_p]
            u_anc      = calibrate_usefulness(raw_u_a, u_all_a, usable_a, usable_all_a, self.coalition_norm)

        loss_anc = compute_feature_loss(logq_anc, ids_a, u_anc, usable_a, loss_kind=self.loss_kind).mean()

        # --- 2. Early-Layer Telescoped Probes (`early_layer .. anchor_layer`, `W_early = 1` at `pos`!) ---
        loss_ear = torch.zeros((), dtype=torch.float32, device=yb.device)
        u_ear_abs = 0.0
        if self.enable_early:
            h_ear_seq = interm[self.early_layer - 1]
            if self.detach_prev and self.early_layer >= 2:
                prev_e = interm[self.early_layer - 2].detach()
                h_ear_seq = prev_e + (h_ear_seq - prev_e)
            x_ear_live = h_ear_seq[b_sel, pos]                  # [B_p, D], grad-attached!

            logq_ear = self.lens_early.log_probs(x_ear_live)
            ids_e, w_e, d_e, val_e = select_diverse_fast(
                logq_ear.detach(), self.lens_early,
                pool_size=self.pool_size, k=k, max_cosine=self.max_cosine, metric=self.metric,
            )

            with torch.no_grad():
                probes_e, usable_e, usable_all_e = suppress_oblique_single_and_joint(
                    x_ear_live.detach(), w_e, d_e, val_e, strength=self.strength, match_norm=self.match_norm
                )  # [B_p, P, D]
                h_win_e = probes_e.unsqueeze(2)                 # [B_p, P, 1, D] — single-token causal trajectory!

                ear_blocks = raw_model.blocks[self.early_layer : self.anchor_layer]
                ear_inputs = [interm[m - 1].detach() for m in range(self.early_layer, self.anchor_layer)]
                kv_ear = extract_clean_kv_for_blocks(ear_blocks, ear_inputs, b_sel, raw_model.cos, raw_model.sin)
                at_anc = advance_local_window_kv(
                    ear_blocks, kv_ear, h_win_e, pos, raw_model.cos, raw_model.sin
                ).squeeze(2)  # [B_p, P, D]

                # Hop-normalize anchor weights so `score_at_anchor` stays in 1-nat units (preventing O(|u_A|^2) decay!)
                if self.hop_norm:
                    norm_a = u_anc.abs().sum(dim=-1, keepdim=True).clamp_min(0.05)
                    u_anc_target = u_anc / norm_a
                else:
                    u_anc_target = u_anc

                at_anc_h  = at_anc.reshape(B_p * P, self.cfg.d_model).float() @ self.lens_anchor.J.t()
                ids_a_exp = ids_a.unsqueeze(1).expand(B_p, P, k).reshape(B_p * P, k)
                u_t_exp   = u_anc_target.unsqueeze(1).expand(B_p, P, k).reshape(B_p * P, k)
                usa_exp   = usable_a.unsqueeze(1).expand(B_p, P, k).reshape(B_p * P, k)
                lq_sel_e  = self.readout.fast_nograd_selected_logprobs(at_anc_h, ids_a_exp)
                scores_e  = compute_feature_loss_pregathered(
                    lq_sel_e, u_t_exp, usa_exp, loss_kind=self.loss_kind
                ).view(B_p, P)

                clean_sc_e = scores_e[:, 0:1]
                raw_u_e    = scores_e[:, 1 : k + 1] - clean_sc_e
                u_all_e    = scores_e[:, k + 1] - scores_e[:, 0]
                u_ear      = calibrate_usefulness(raw_u_e, u_all_e, usable_e, usable_all_e, self.coalition_norm)
                u_ear_abs  = float(u_ear.abs().mean().item())

            loss_ear = compute_feature_loss(logq_ear, ids_e, u_ear, usable_e, loss_kind=self.loss_kind).mean()

        tot_aux = 0.5 * (loss_anc + loss_ear) if self.enable_early else loss_anc
        stats = {
            "u_anc_abs": round(float(u_anc.abs().mean().item()), 5),
            "u_anc_pos_frac": round(float((u_anc > 0).float().mean().item()), 3),
            "u_ear_abs": round(u_ear_abs, 5),
            "loss_anc": round(float(loss_anc.detach().item()), 5),
            "loss_ear": round(float(loss_ear.detach().item()), 5),
        }
        return tot_aux, tot_aux.detach(), stats
