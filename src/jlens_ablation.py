"""
Causal-Ablation Workspace `J-Lens` (`jlens_mode = "ablation"`):
  1. Supports multi-position `--abl-active-frac` (e.g. 0.5%, 2%, 6%, 15% of all batch tokens, sampled uniformly in a 2:1
     ratio from the batch-wide Top-10% Hardest and Bottom-5% Easiest pools) sharing a SINGLE clean `(K, V)` cache per sequence!
  2. Reads clean intermediate hidden states `h_l` through a frozen `J-Lens` snapshot (`J_l`, activation covariance `Sigma_l`,
     and frozen `[out_norm, tok_emb]` softcapped readout) via zero-HBM-spill `SelectedSoftcapLinearLogProbsFn`.
  3. Greedily selects up to `k` (`8` or `16`) diverse active token hypotheses from `top-64` (`|cos| <= max_cosine`).
  4. Suppresses each candidate direction via covariance-matched oblique projection `d_i = Sigma_l @ (J_l^T U_i)` plus 1
     joint all-`k` suppression probe `U_all` (`(W^T D + eps I)^{-1} W^T x`).
  5. Evaluates signed causal usefulness `u_i = score(ablated_i) - score(clean)` under `no_grad` via a custom Triton
     Flash-Prefix + Local-Causal KV-window kernel (`[p .. p + W - 1]`), cutting probe FLOPs & HBM traffic massively:
       - At `anchor_layer` (`L09 -> L12`): `score` is downstream gold softcapped NLL over the `W`-token window.
       - At `early_layer`  (`L03 -> L09`): `score` stops at `anchor_layer` (`W=1` at `p`) against hop-normalized `u_anc`.
  6. Differentiates ONLY the clean `log_probs` path (`stopgrad` on all probe usefulnesses), supporting
     `superpos_unlikelihood` (default), `unlikelihood`, and `signed_raw`.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import triton
import triton.language as tl

from model_70m import apply_rope
from fast_kernels import (
    _softcap_lse_2half_reg_kernel,
    _softcap_target_nll_2half_reg_kernel,
    extract_clean_kv_fast,
)
from ultra_probe_kernels import advance_local_window_kv_ultra

RHO_HARMONIC = 3.0 / math.sqrt(5.0 * math.pi)  # 0.75694...
EOT_TOKEN    = 50256


# ==============================================================================
# 1. Custom ROCm Triton Kernels: Online-LSE, Selected-LogProbs Bwd, Greedy Diversity, & Flash-Prefix KV
# ==============================================================================

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
def _softcap_selected_base_bwd_kernel(
    logits_ptr, stride_m, lse_ptr, sum_g_ptr,
    V: tl.constexpr, SOFTCAP: tl.constexpr, INV_SOFTCAP: tl.constexpr, BLOCK_V: tl.constexpr,
):
    """Overwrites `logits_ptr` [N, V] in-place with `-sum_g[r] * softmax(lg)[r, v] * ds[r, v]` in BF16."""
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * stride_m
    lse = tl.load(lse_ptr + row)
    sg  = tl.load(sum_g_ptr + row)
    LOG2E: tl.constexpr = 1.4426950408889634
    for start_v in range(0, V, BLOCK_V):
        offs_v = start_v + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V
        raw = tl.load(row_ptr + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        if SOFTCAP > 0.0:
            z = tl.minimum(tl.maximum(raw * INV_SOFTCAP, -20.0), 20.0)
            e2z = tl.math.exp2(2.0 * LOG2E * z)
            th = (e2z - 1.0) / (e2z + 1.0)
            lg = SOFTCAP * th
            ds = 1.0 - th * th
        else:
            lg = raw
            ds = 1.0
        prob = tl.math.exp2((lg - lse) * LOG2E)
        grad = (-sg * prob) * ds
        tl.store(row_ptr + offs_v, grad.to(tl.bfloat16), mask=mask_v)


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


@triton.jit
def _flash_prefix_kv_kernel(
    q_ptr, k_ptr, v_ptr, kv_slot_ptr, pos_ptr,
    o_ptr, lse_ptr,
    stride_qn, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kt,
    stride_on, stride_oh, stride_om,
    stride_ln, stride_lh,
    PW, SM_SCALE,
    BLOCK_PW: tl.constexpr, BLOCK_T: tl.constexpr, DH: tl.constexpr,
):
    """
    Exact FlashAttention prefix summary over `0 .. pos[n] - 1` sharing per-sequence `K_clean[kv_slot[n]], V_clean[kv_slot[n]]`.
    Zero sequence padding, skips `t >= pos[n]`, and allocates zero `[N, H, PW, T]` HBM attention matrix!
    """
    pid_n = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)

    p_len = tl.load(pos_ptr + pid_n)
    offs_m = tl.arange(0, BLOCK_PW)
    offs_d = tl.arange(0, DH)
    mask_m = offs_m < PW

    if p_len <= 0:
        o_base = o_ptr + pid_n * stride_on + pid_h * stride_oh
        tl.store(o_base + offs_m[:, None] * stride_om + offs_d[None, :], 0.0, mask=mask_m[:, None])
        tl.store(lse_ptr + pid_n * stride_ln + pid_h * stride_lh + offs_m, -float("inf"), mask=mask_m)
        return

    b_slot = tl.load(kv_slot_ptr + pid_n).to(tl.int64)
    q_base = q_ptr + pid_n * stride_qn + pid_h * stride_qh
    k_base = k_ptr + b_slot * stride_kb + pid_h * stride_kh
    v_base = v_ptr + b_slot * stride_kb + pid_h * stride_kh

    q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :], mask=mask_m[:, None], other=0.0).to(tl.bfloat16)
    LOG2E: tl.constexpr = 1.4426950408889634
    qk_scale = SM_SCALE * LOG2E

    m_i = tl.full((BLOCK_PW,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_PW,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_PW, DH), dtype=tl.float32)

    for t0 in range(0, p_len, BLOCK_T):
        offs_t = t0 + tl.arange(0, BLOCK_T)
        mask_t = offs_t < p_len
        k_tile = tl.load(k_base + offs_t[:, None] * stride_kt + offs_d[None, :], mask=mask_t[:, None], other=0.0).to(tl.bfloat16)
        qk = tl.dot(q, tl.trans(k_tile)) * qk_scale
        qk = tl.where(mask_t[None, :], qk, -float("inf"))

        blk_max = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, blk_max)
        alpha = tl.math.exp2(m_i - m_new)
        p_w = tl.math.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p_w, axis=1)
        acc = acc * alpha[:, None]

        v_tile = tl.load(v_base + offs_t[:, None] * stride_kt + offs_d[None, :], mask=mask_t[:, None], other=0.0).to(tl.bfloat16)
        acc += tl.dot(p_w.to(tl.bfloat16), v_tile)
        m_i = m_new

    inv_l = 1.0 / tl.maximum(l_i, 1e-30)
    o_out = acc * inv_l[:, None]
    lse_out = m_i * 0.6931471805599453 + tl.math.log(tl.maximum(l_i, 1e-30))

    o_base = o_ptr + pid_n * stride_on + pid_h * stride_oh
    tl.store(o_base + offs_m[:, None] * stride_om + offs_d[None, :], o_out, mask=mask_m[:, None])
    tl.store(lse_ptr + pid_n * stride_ln + pid_h * stride_lh + offs_m, lse_out, mask=mask_m)


class _SelectedSoftcapLinearLogProbsFn(torch.autograd.Function):
    """
    Zero-HBM-spill differentiable `log_softmax(softcap * tanh((hn @ U.T) / softcap)).gather(-1, ids)` `[N, k]`.
    Saves only `hn` [N, D] + `lse` [N] (80 MB at N=40,000 instead of 48 GB!), running BF16 MFMA GEMMs in chunks.
    """
    @staticmethod
    def forward(ctx, hn_fp32: Tensor, U_bf16: Tensor, ids: Tensor, softcap: float, chunk_sz: int = 4096) -> Tensor:
        N, D = hn_fp32.shape
        k = ids.shape[1]
        V = U_bf16.shape[0]
        inv_sc = (1.0 / softcap) if softcap > 0.0 else 0.0
        use_2half = (32768 < V <= 65536)

        lq_sel = torch.empty((N, k), dtype=torch.float32, device=hn_fp32.device)
        lse_all = torch.empty((N,), dtype=torch.float32, device=hn_fp32.device)
        buf = torch.empty((min(N, chunk_sz), V), dtype=torch.bfloat16, device=hn_fp32.device)

        for s in range(0, N, chunk_sz):
            e = min(s + chunk_sz, N)
            nr = e - s
            hn_bf = hn_fp32[s:e].detach().to(torch.bfloat16)
            b_sl = buf[:nr]
            l_sl = lse_all[s:e]
            torch.matmul(hn_bf, U_bf16.t(), out=b_sl)
            if use_2half:
                _softcap_lse_2half_reg_kernel[(nr,)](
                    b_sl, b_sl.stride(0), l_sl, V=V, SOFTCAP=float(softcap), INV_SOFTCAP=float(inv_sc), HALF_V=32768, num_warps=8
                )
            else:
                _softcap_lse_nograd_kernel[(nr,)](
                    b_sl, b_sl.stride(0), l_sl, V=V, SOFTCAP=float(softcap), INV_SOFTCAP=float(inv_sc), BLOCK_V=4096, num_warps=8
                )
            raw_k = b_sl.gather(1, ids[s:e]).float()
            lg_k = (softcap * torch.tanh(raw_k * inv_sc)) if softcap > 0.0 else raw_k
            lq_sel[s:e] = lg_k - l_sl.unsqueeze(1)

        ctx.save_for_backward(hn_fp32.detach(), U_bf16, ids, lse_all)
        ctx.softcap = float(softcap)
        ctx.chunk_sz = int(chunk_sz)
        return lq_sel

    @staticmethod
    def backward(ctx, grad_lq: Tensor):
        hn_saved, U_bf16, ids, lse_all = ctx.saved_tensors
        softcap = ctx.softcap
        chunk_sz = ctx.chunk_sz
        inv_sc = (1.0 / softcap) if softcap > 0.0 else 0.0

        N, D = hn_saved.shape
        V = U_bf16.shape[0]
        dhn_bf = torch.empty((N, D), dtype=torch.bfloat16, device=hn_saved.device)
        buf = torch.empty((min(N, chunk_sz), V), dtype=torch.bfloat16, device=hn_saved.device)

        for s in range(0, N, chunk_sz):
            e = min(s + chunk_sz, N)
            nr = e - s
            hn_bf = hn_saved[s:e].to(torch.bfloat16)
            b_sl = buf[:nr]
            torch.matmul(hn_bf, U_bf16.t(), out=b_sl)

            ids_sl = ids[s:e]
            raw_k = b_sl.gather(1, ids_sl).float()
            if softcap > 0.0:
                th_k = torch.tanh(raw_k * inv_sc)
                ds_k = 1.0 - th_k * th_k
            else:
                ds_k = torch.ones_like(raw_k)

            g_k = grad_lq[s:e].float()
            sum_g = g_k.sum(dim=-1).contiguous()
            _softcap_selected_base_bwd_kernel[(nr,)](
                b_sl, b_sl.stride(0), lse_all[s:e], sum_g,
                V=V, SOFTCAP=softcap, INV_SOFTCAP=inv_sc, BLOCK_V=4096, num_warps=8,
            )
            b_sl.scatter_add_(1, ids_sl, (g_k * ds_k).to(torch.bfloat16))
            torch.matmul(b_sl, U_bf16, out=dhn_bf[s:e])

        return dhn_bf.to(hn_saved.dtype), None, None, None, None


# ==============================================================================
# 2. Readout, Workspace J-Lens, Diversity Filter, Oblique Suppression, & Losses
# ==============================================================================

def pure_rms_norm_fp32(x: Tensor, eps: float = 1e-6) -> Tensor:
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)


class FrozenSoftcapReadout(nn.Module):
    """Frozen snapshot of final RMSNorm + tied unembedding + logit softcap."""

    def __init__(self, norm_weight: Tensor, unemb_weight: Tensor, softcap: float = 30.0, eps: float = 1e-6):
        super().__init__()
        self.softcap = float(softcap)
        self.eps     = float(eps)
        self.register_buffer("norm_w", norm_weight.detach().float().clone())
        self.register_buffer("U", unemb_weight.detach().float().clone())
        self.register_buffer("U_bf16", unemb_weight.detach().to(torch.bfloat16).clone())

    @torch.no_grad()
    def copy_from_(self, norm_weight: Tensor, unemb_weight: Tensor):
        self.update_ema_(norm_weight, unemb_weight, ema=0.0)

    @torch.no_grad()
    def update_ema_(self, norm_weight: Tensor, unemb_weight: Tensor, ema: float = 0.0):
        nw = norm_weight.detach().float()
        uw = unemb_weight.detach().float()
        if ema <= 0.0:
            self.norm_w.copy_(nw)
            self.U.copy_(uw)
        else:
            self.norm_w.mul_(ema).add_(nw, alpha=(1.0 - ema))
            self.U.mul_(ema).add_(uw, alpha=(1.0 - ema))
        self.U_bf16.copy_(self.U.to(torch.bfloat16))

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
    def fast_topk_pool_ids(self, h_2d: Tensor, pool_size: int = 64, chunk_sz: int = 16384) -> Tensor:
        """Finds top-`pool_size` token IDs `[N, M]` without allocating `[N, V]` FP32 tensors."""
        M_pool = min(pool_size, self.U_bf16.shape[0])
        if not h_2d.is_cuda or self.U_bf16.shape[0] < 1024:
            return self.log_probs(h_2d).topk(M_pool, dim=-1).indices
        N, D = h_2d.shape
        V = self.U_bf16.shape[0]
        from fast_kernels import FusedRMSNormFn
        hn_all = FusedRMSNormFn.apply(h_2d.to(torch.bfloat16), self.norm_w, self.eps) if D == 512 else None
        out_ids = torch.empty((N, M_pool), dtype=torch.long, device=h_2d.device)
        buf = torch.empty((min(N, chunk_sz), V), dtype=torch.bfloat16, device=h_2d.device)
        for s in range(0, N, chunk_sz):
            e = min(s + chunk_sz, N)
            nr = e - s
            if hn_all is not None:
                hn = hn_all[s:e]
            else:
                hf = h_2d[s:e].float()
                hn = (hf * torch.rsqrt(hf.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.norm_w).to(torch.bfloat16)
            b_sl = buf[:nr]
            torch.matmul(hn, self.U_bf16.t(), out=b_sl)
            out_ids[s:e] = b_sl.topk(M_pool, dim=-1).indices
        return out_ids

    def selected_log_probs(self, h_2d: Tensor, ids_2d: Tensor) -> Tensor:
        """Differentiable `log_probs(h_2d).gather(-1, ids_2d)` `[N, k]` with O(N*D) saved memory."""
        if not h_2d.is_cuda or self.U_bf16.shape[0] < 1024:
            return self.log_probs(h_2d).gather(-1, ids_2d)
        with torch.amp.autocast(device_type=h_2d.device.type, enabled=False):
            hf = h_2d.float()
            hn = hf * torch.rsqrt(hf.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.norm_w
            return _SelectedSoftcapLinearLogProbsFn.apply(hn, self.U_bf16, ids_2d, self.softcap, 16384)

    @torch.no_grad()
    def fast_nograd_target_nll(self, h_2d: Tensor, targets_1d: Tensor, chunk_sz: int = 32768) -> Tensor:
        if not h_2d.is_cuda or self.U_bf16.shape[0] < 1024:
            return F.cross_entropy(self.logits(h_2d), targets_1d, reduction="none")
        N, D = h_2d.shape
        V = self.U_bf16.shape[0]
        inv_sc = (1.0 / self.softcap) if self.softcap > 0.0 else 0.0
        use_2half = (32768 < V <= 65536)
        from fast_kernels import FusedRMSNormFn
        hn_all = FusedRMSNormFn.apply(h_2d.to(torch.bfloat16), self.norm_w, self.eps) if D == 512 else None
        out_nll = torch.empty((N,), dtype=torch.float32, device=h_2d.device)
        buf = torch.empty((min(N, chunk_sz), V), dtype=torch.bfloat16, device=h_2d.device)
        lse = None if use_2half else torch.empty((min(N, chunk_sz),), dtype=torch.float32, device=h_2d.device)
        for s in range(0, N, chunk_sz):
            e = min(s + chunk_sz, N)
            nr = e - s
            if hn_all is not None:
                hn = hn_all[s:e]
            else:
                hf = h_2d[s:e].float()
                hn = (hf * torch.rsqrt(hf.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.norm_w).to(torch.bfloat16)
            b_sl = buf[:nr]
            torch.matmul(hn, self.U_bf16.t(), out=b_sl)
            if use_2half:
                _softcap_target_nll_2half_reg_kernel[(nr,)](
                    b_sl, b_sl.stride(0), targets_1d[s:e], out_nll[s:e],
                    V=V, SOFTCAP=self.softcap, INV_SOFTCAP=inv_sc, HALF_V=32768, num_warps=8,
                )
            else:
                l_sl = lse[:nr]
                _softcap_lse_nograd_kernel[(nr,)](
                    b_sl, b_sl.stride(0), l_sl, V=V, SOFTCAP=self.softcap, INV_SOFTCAP=inv_sc, BLOCK_V=4096, num_warps=8
                )
                raw_t = b_sl.gather(1, targets_1d[s:e].unsqueeze(1)).squeeze(1).float()
                lg_t = (self.softcap * torch.tanh(raw_t * inv_sc)) if self.softcap > 0.0 else raw_t
                out_nll[s:e] = l_sl - lg_t
        return out_nll

    @torch.no_grad()
    def fast_nograd_selected_logprobs(self, h_2d: Tensor, sel_ids_2d: Tensor, chunk_sz: int = 16384) -> Tensor:
        if not h_2d.is_cuda or self.U_bf16.shape[0] < 1024:
            return self.log_probs(h_2d).gather(-1, sel_ids_2d)
        N, D = h_2d.shape
        k = sel_ids_2d.shape[1]
        V = self.U_bf16.shape[0]
        inv_sc = (1.0 / self.softcap) if self.softcap > 0.0 else 0.0
        use_2half = (32768 < V <= 65536)
        from fast_kernels import FusedRMSNormFn
        hn_all = FusedRMSNormFn.apply(h_2d.to(torch.bfloat16), self.norm_w, self.eps) if D == 512 else None
        out_lq = torch.empty((N, k), dtype=torch.float32, device=h_2d.device)
        buf = torch.empty((min(N, chunk_sz), V), dtype=torch.bfloat16, device=h_2d.device)
        lse = torch.empty((min(N, chunk_sz),), dtype=torch.float32, device=h_2d.device)
        for s in range(0, N, chunk_sz):
            e = min(s + chunk_sz, N)
            nr = e - s
            if hn_all is not None:
                hn = hn_all[s:e]
            else:
                hf = h_2d[s:e].float()
                hn = (hf * torch.rsqrt(hf.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.norm_w).to(torch.bfloat16)
            b_sl = buf[:nr]
            l_sl = lse[:nr]
            torch.matmul(hn, self.U_bf16.t(), out=b_sl)
            if use_2half:
                _softcap_lse_2half_reg_kernel[(nr,)](
                    b_sl, b_sl.stride(0), l_sl, V=V, SOFTCAP=self.softcap, INV_SOFTCAP=inv_sc, HALF_V=32768, num_warps=8
                )
            else:
                _softcap_lse_nograd_kernel[(nr,)](
                    b_sl, b_sl.stride(0), l_sl, V=V, SOFTCAP=self.softcap, INV_SOFTCAP=inv_sc, BLOCK_V=4096, num_warps=8
                )
            raw_k = b_sl.gather(1, sel_ids_2d[s:e]).float()
            lg_k = (self.softcap * torch.tanh(raw_k * inv_sc)) if self.softcap > 0.0 else raw_k
            out_lq[s:e] = lg_k - l_sl.unsqueeze(1)
        return out_lq


class WorkspaceJLens(nn.Module):
    """Layer lens `J_l` (`h_final ~ pure_rms_norm_fp32(x) @ J.T`) + running covariance `Sigma_l` + shared frozen readout."""

    def __init__(self, d_model: int, readout: FrozenSoftcapReadout, device: torch.device, norm_input: bool = True):
        super().__init__()
        self.d_model = d_model
        self.readout = readout
        self.norm_input = norm_input
        self.register_buffer("J", torch.eye(d_model, dtype=torch.float32, device=device))
        self.register_buffer("Sigma", torch.eye(d_model, dtype=torch.float32, device=device))

    def project_hidden(self, x: Tensor) -> Tensor:
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            xf = pure_rms_norm_fp32(x) if self.norm_input else x.float()
            return xf @ self.J.t()

    def log_probs(self, x: Tensor) -> Tensor:
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            return self.readout.log_probs(self.project_hidden(x))

    @torch.no_grad()
    def topk_pool_ids(self, x: Tensor, pool_size: int = 64) -> Tensor:
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            return self.readout.fast_topk_pool_ids(self.project_hidden(x.detach()), pool_size=pool_size)

    def selected_log_probs(self, x: Tensor, ids: Tensor) -> Tensor:
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            return self.readout.selected_log_probs(self.project_hidden(x), ids)

    @torch.no_grad()
    def candidate_vectors(self, ids: Tensor, metric: str = "cov") -> tuple[Tensor, Tensor, Tensor]:
        with torch.amp.autocast(device_type=ids.device.type, enabled=False):
            u_tok  = self.readout.U[ids]
            w_read = torch.matmul(u_tok, self.J)
            if metric == "cov":
                d_write = torch.matmul(w_read, self.Sigma)
                denom = (w_read * d_write).sum(dim=-1, keepdim=True).clamp_min(1e-16).sqrt()
                u_norm = u_tok.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                u_sim  = u_tok / u_norm
                u_sim  = torch.where(denom > 1e-8, u_sim, torch.zeros_like(u_sim))
            else:
                d_write = w_read
                w_norm  = w_read.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                u_sim   = torch.where(w_read.norm(dim=-1, keepdim=True) > 1e-8, w_read / w_norm, torch.zeros_like(w_read))
            return w_read, d_write, u_sim


@torch.no_grad()
def select_diverse_from_pool(
    pool: Tensor,
    lens: WorkspaceJLens,
    k: int = 8,
    max_cosine: float = 0.85,
    metric: str = "cov",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    M_pool = pool.shape[-1]
    with torch.amp.autocast(device_type=pool.device.type, enabled=False):
        w_pool, d_pool, u_sim = lens.candidate_vectors(pool, metric=metric)
        B, _, D = w_pool.shape
        sim_norms = u_sim.float().norm(dim=-1)
        cos_mat   = torch.matmul(u_sim.float(), u_sim.float().transpose(-1, -2)).abs()

        if pool.is_cuda and M_pool <= 64:
            acc_i8 = torch.empty((B, M_pool), dtype=torch.int8, device=pool.device)
            _greedy_diverse_accept_kernel[(B,)](
                sim_norms.contiguous(), cos_mat.contiguous(), acc_i8,
                M_pool, k, float(max_cosine), BLOCK_M=64, num_warps=2,
            )
            accepted = acc_i8.bool()
        else:
            accepted = torch.zeros((B, M_pool), dtype=torch.bool, device=pool.device)
            n_acc    = torch.zeros((B,), dtype=torch.long, device=pool.device)
            for j in range(M_pool):
                ok_len = sim_norms[:, j] > 0.5
                if j == 0:
                    conflict = torch.zeros((B,), dtype=torch.bool, device=pool.device)
                else:
                    prev_acc = accepted[:, :j]
                    c_prev   = cos_mat[:, j, :j]
                    conflict = ((c_prev > max_cosine) & prev_acc).any(dim=-1)
                take = ok_len & (~conflict) & (n_acc < k)
                accepted[:, j] = take
                n_acc = n_acc + take.long()

        slot_idx = torch.cumsum(accepted.long(), dim=-1) - 1
        ids_out  = torch.zeros((B, k), dtype=torch.long, device=pool.device)
        w_out    = torch.zeros((B, k, D), dtype=torch.float32, device=pool.device)
        d_out    = torch.zeros((B, k, D), dtype=torch.float32, device=pool.device)
        val_out  = torch.zeros((B, k), dtype=torch.bool, device=pool.device)

        b_idx, m_idx = torch.nonzero(accepted, as_tuple=True)
        if b_idx.numel() > 0:
            s_idx = slot_idx[b_idx, m_idx]
            ids_out[b_idx, s_idx] = pool[b_idx, m_idx]
            w_out[b_idx, s_idx]   = w_pool[b_idx, m_idx]
            d_out[b_idx, s_idx]   = d_pool[b_idx, m_idx]
            val_out[b_idx, s_idx] = True

        return ids_out, w_out, d_out, val_out


@torch.no_grad()
def select_diverse_fast(
    logq: Tensor,
    lens: WorkspaceJLens,
    pool_size: int = 64,
    k: int = 8,
    max_cosine: float = 0.85,
    metric: str = "cov",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    M_pool = min(pool_size, logq.shape[-1])
    pool = logq.float().topk(M_pool, dim=-1).indices
    return select_diverse_from_pool(pool, lens, k=k, max_cosine=max_cosine, metric=metric)


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

_compiled_suppress_oblique = torch.compile(suppress_oblique_single_and_joint, dynamic=True)


def compute_feature_loss_pregathered(
    lq_sel: Tensor,
    weights: Tensor,
    valid: Tensor,
    loss_kind: str = "superpos_unlikelihood",
    exp_beta: float = 0.0,
    exp_norm: bool = True,
) -> Tensor:
    with torch.amp.autocast(device_type=lq_sel.device.type, enabled=False):
        w = torch.where(valid, weights.detach().float(), torch.zeros_like(weights, dtype=torch.float32))
        lq_f = lq_sel.float()

        if loss_kind == "signed_raw":
            return -(w * lq_f).sum(dim=-1)

        if loss_kind == "exp_signed_clamped":
            u_c = torch.where(valid, w.clamp(-0.6, 0.6), torch.zeros_like(w))
            if exp_beta > 0.0:
                u_raw = torch.sign(u_c) * (torch.expm1(exp_beta * u_c.abs()) * (1.0 / exp_beta))
                u_raw = torch.where(valid, u_raw, torch.zeros_like(u_raw))
                if exp_norm:
                    n_val = valid.float().sum().clamp_min(1.0)
                    mean_c = u_c.abs().sum() / n_val
                    mean_r = u_raw.abs().sum() / n_val
                    u_tilde = u_raw * (mean_c / mean_r.clamp_min(1e-8))
                else:
                    u_tilde = u_raw
            else:
                u_tilde = u_c

            w_pos = F.relu(u_tilde)
            w_neg = F.relu(-u_tilde)
            q_sel = lq_f.exp().clamp(0.0, 1.0 - 1e-7)
            pos_mask = (u_tilde > 0.0) & valid
            sum_q_pos = (q_sel * pos_mask.float()).sum(dim=-1, keepdim=True)
            q_bg = (1.0 - sum_q_pos).clamp_min(1e-6)
            log_q_bg  = torch.log(q_bg)
            log_p_ind = lq_f - torch.logaddexp(lq_f, log_q_bg.expand_as(lq_f))
            pos_term  = -(w_pos * log_p_ind).sum(dim=-1)
            # Full -(1 - q_i) suppression strength down to log_q = -9.0 (q ~ 1.23e-4), zero grad / zero -inf blowup below!
            neg_term  = (w_neg * lq_f.clamp(min=-9.0)).sum(dim=-1)
            return pos_term + neg_term

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
    exp_beta: float = 0.0,
    exp_norm: bool = True,
) -> Tensor:
    lq_sel = logq.float().gather(-1, ids)
    return compute_feature_loss_pregathered(
        lq_sel, weights, valid, loss_kind=loss_kind, exp_beta=exp_beta, exp_norm=exp_norm
    )


# ==============================================================================
# 3. Shared-Sequence KV Extraction & Flash-Prefix Multi-Position KV-Window Propagation
# ==============================================================================

@torch.no_grad()
def extract_clean_kv_for_blocks(
    blocks: nn.ModuleList,
    clean_inputs: list[Tensor],
    b_indices: Tensor,
    cos_full: Tensor,
    sin_full: Tensor,
) -> list[tuple[Tensor, Tensor]]:
    """Computes clean `(K, V)` caches `[B_u, H, T, Dh]` ONCE per unique sequence in `b_indices`."""
    kv_list = []
    T = clean_inputs[0].shape[1]
    cos_t, sin_t = cos_full[:, :, :T, :], sin_full[:, :, :T, :]
    for blk, x_in in zip(blocks, clean_inputs):
        xb = x_in[b_indices]
        B_u, _, D = xb.shape
        H, Dh = blk.cfg.n_heads, blk.cfg.head_dim
        qkv = blk.wqkv(blk.attn_norm(xb)).view(B_u, T, 3, H, Dh)
        k_c = apply_rope(blk.k_norm(qkv[:, :, 1]).transpose(1, 2), cos_t, sin_t).contiguous()
        v_c = qkv[:, :, 2].transpose(1, 2).contiguous()
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
    kv_slot: Tensor | None = None,
) -> Tensor:
    """
    Exact KV-cached causal window propagation supporting arbitrary multi-position `kv_slot` (`[N_pos] -> [0..B_u-1]`)!
    Uses our ROCm Triton `_flash_prefix_kv_kernel` when `Dh == 64` on GPU (zero HBM attention matrix, skips `t >= pos`),
    with exact PyTorch fallback for arbitrary CPU/small-Dh unit tests.
    """
    N_pos, P, W, D = h_win.shape
    T = kv_list[0][0].shape[2]
    H, Dh = blocks[0].cfg.n_heads, blocks[0].cfg.head_dim
    inv_sqrt_dh = 1.0 / math.sqrt(float(Dh))
    if kv_slot is None:
        kv_slot = torch.arange(N_pos, device=h_win.device, dtype=torch.int32)
    else:
        kv_slot = kv_slot.to(torch.int32).contiguous()

    pos_i32 = positions.to(torch.int32).contiguous()
    w_offs = torch.arange(W, device=h_win.device)
    win_pos = positions[:, None] + w_offs[None, :]              # [N_pos, W]
    cos_Sq  = cos_full[0, 0][win_pos][:, None, None, :, :]      # [N_pos, 1, 1, W, Dh//2]
    sin_Sq  = sin_full[0, 0][win_pos][:, None, None, :, :]
    loc_causal = torch.ones((W, W), dtype=torch.bool, device=h_win.device).tril(0)

    PW = P * W
    use_triton_flash = h_win.is_cuda and (Dh == 64)
    if use_triton_flash:
        block_pw = max(16, triton.next_power_of_2(PW))
        o_pre_buf   = torch.empty((N_pos, H, PW, Dh), dtype=torch.float32, device=h_win.device)
        lse_pre_buf = torch.empty((N_pos, H, PW), dtype=torch.float32, device=h_win.device)
    else:
        t_ar = torch.arange(T, device=h_win.device)[None, :]
        pre_valid = (t_ar < positions[:, None])

    x = h_win
    for blk, (k_clean, v_clean) in zip(blocks, kv_list):
        x_flat = x.reshape(N_pos * P, W, D)
        qkv = blk.wqkv(blk.attn_norm(x_flat)).view(N_pos, P, W, 3, H, Dh)
        q_w = apply_rope(blk.q_norm(qkv[:, :, :, 0]).transpose(2, 3), cos_Sq, sin_Sq)  # [N_pos, P, H, W, Dh]
        k_w = apply_rope(blk.k_norm(qkv[:, :, :, 1]).transpose(2, 3), cos_Sq, sin_Sq)
        v_w = qkv[:, :, :, 2].transpose(2, 3)

        if use_triton_flash:
            q_pw = q_w.permute(0, 2, 1, 3, 4).reshape(N_pos, H, PW, Dh).to(torch.bfloat16).contiguous()
            k_bf = k_clean if k_clean.dtype == torch.bfloat16 else k_clean.to(torch.bfloat16).contiguous()
            v_bf = v_clean if v_clean.dtype == torch.bfloat16 else v_clean.to(torch.bfloat16).contiguous()
            _flash_prefix_kv_kernel[(N_pos, H)](
                q_pw, k_bf, v_bf, kv_slot, pos_i32,
                o_pre_buf, lse_pre_buf,
                q_pw.stride(0), q_pw.stride(1), q_pw.stride(2),
                k_bf.stride(0), k_bf.stride(1), k_bf.stride(2),
                o_pre_buf.stride(0), o_pre_buf.stride(1), o_pre_buf.stride(2),
                lse_pre_buf.stride(0), lse_pre_buf.stride(1),
                PW=PW, SM_SCALE=inv_sqrt_dh,
                BLOCK_PW=block_pw, BLOCK_T=64, DH=64, num_warps=4,
            )
            o_pre_p   = o_pre_buf.view(N_pos, H, P, W, Dh).permute(0, 2, 1, 3, 4)    # [N_pos, P, H, W, Dh]
            lse_pre_p = lse_pre_buf.view(N_pos, H, P, W).permute(0, 2, 1, 3)         # [N_pos, P, H, W]

            with torch.amp.autocast(device_type=x.device.type, enabled=False):
                sc_loc  = torch.matmul(q_w.float(), k_w.float().transpose(-1, -2)) * inv_sqrt_dh
                sc_loc  = sc_loc.masked_fill(~loc_causal[None, None, None, :, :], -float("inf"))
                lse_loc = torch.logsumexp(sc_loc, dim=-1)
                pr_loc  = torch.exp(sc_loc - lse_loc.unsqueeze(-1))
                out_loc = torch.matmul(pr_loc, v_w.float())

                lse_tot = torch.logaddexp(lse_pre_p, lse_loc)
                w_pre   = torch.exp(lse_pre_p - lse_tot).unsqueeze(-1)
                w_loc   = torch.exp(lse_loc - lse_tot).unsqueeze(-1)
                y_attn  = (w_pre * o_pre_p + w_loc * out_loc).to(x.dtype).transpose(2, 3).reshape(N_pos * P, W, D)
        else:
            k_sel = k_clean[kv_slot.long()]
            v_sel = v_clean[kv_slot.long()]
            q_pw  = q_w.permute(0, 2, 1, 3, 4).reshape(N_pos, H, PW, Dh)
            sc_pre = torch.matmul(q_pw.float(), k_sel.float().transpose(-1, -2)) * inv_sqrt_dh
            sc_pre = sc_pre.masked_fill(~pre_valid[:, None, None, :], -float("inf"))
            sc_pre = sc_pre.view(N_pos, H, P, W, T).permute(0, 2, 1, 3, 4)

            sc_loc = torch.matmul(q_w.float(), k_w.float().transpose(-1, -2)) * inv_sqrt_dh
            sc_loc = sc_loc.masked_fill(~loc_causal[None, None, None, :, :], -float("inf"))

            probs_all = torch.softmax(torch.cat([sc_pre, sc_loc], dim=-1), dim=-1).to(x.dtype)
            pr_pre = probs_all[..., :T].permute(0, 2, 1, 3, 4).reshape(N_pos, H, PW, T)
            pr_loc = probs_all[..., T:]
            out_pre = torch.matmul(pr_pre, v_sel.to(x.dtype)).view(N_pos, H, P, W, Dh).permute(0, 2, 1, 3, 4)
            out_loc = torch.matmul(pr_loc, v_w.to(x.dtype))
            y_attn  = (out_pre + out_loc).transpose(2, 3).reshape(N_pos * P, W, D)

        x_flat = x_flat + blk.wo(y_attn)
        g, u   = blk.wgu(blk.mlp_norm(x_flat)).chunk(2, dim=-1)
        x_flat = x_flat + blk.wdown(F.silu(g) * u)
        x = x_flat.view(N_pos, P, W, D)

    return x


@torch.no_grad()
def calibrate_usefulness(
    raw_u: Tensor,
    u_all: Tensor,
    valid: Tensor,
    usable_all: Tensor,
    coalition_norm: bool = True,
) -> Tensor:
    u = torch.where(valid, raw_u, torch.zeros_like(raw_u))
    if not coalition_norm:
        return u
    s_ind  = u.abs().sum(dim=-1, keepdim=True)
    target = torch.maximum(u_all.abs().unsqueeze(-1), 0.25 * s_ind)
    gamma  = (target / s_ind.clamp_min(1e-6)).clamp(0.25, 4.0)
    gamma  = torch.where((s_ind > 1e-6) & usable_all.unsqueeze(-1), gamma, torch.ones_like(gamma))
    return u * gamma


# ==============================================================================
# 4. Main Causal-Ablation Workspace J-Lens Controller
# ==============================================================================

class CausalAblationJLensController:
    def __init__(
        self,
        raw_model: nn.Module,
        device: torch.device,
        early_layer: int = 3,
        anchor_layer: int = 9,
        active_frac: float = 0.02,
        probe_seqs: int = 64,
        pool_size: int = 64,
        k_dirs: int = 8,
        window: int = 4,
        max_cosine: float = 0.85,
        strength: float = 1.0,
        match_norm: bool = True,
        metric: str = "cov",
        loss_kind: str = "superpos_unlikelihood",
        exp_beta: float = 0.0,
        exp_norm: bool = True,
        coalition_norm: bool = True,
        hop_norm: bool = True,
        enable_early: bool = True,
        layer_mode: str = "fixed",
        pos_mode: str = "hard_easy",
        detach_prev: bool = False,
        ridge: float = 0.10,
        ema: float = 0.50,
        pos_chunk_sz: int = 8192,
        seed: int = 42,
    ):
        assert 1 <= early_layer < anchor_layer < raw_model.cfg.n_layers
        assert layer_mode in ("fixed", "random_gap4", "random_anc_7_10")
        self.cfg            = raw_model.cfg
        self.device         = device
        self.early_layer    = early_layer
        self.anchor_layer   = anchor_layer
        self.layer_mode     = layer_mode
        self.active_frac    = float(active_frac)
        self.probe_seqs     = probe_seqs
        self.pool_size      = pool_size
        self.k_dirs         = k_dirs
        self.window         = window
        self.max_cosine     = max_cosine
        self.strength       = strength
        self.match_norm     = match_norm
        self.metric         = metric
        self.loss_kind      = loss_kind
        self.exp_beta       = float(exp_beta)
        self.exp_norm       = bool(exp_norm)
        self.coalition_norm = coalition_norm
        self.hop_norm       = hop_norm
        self.enable_early   = enable_early
        self.pos_mode       = pos_mode
        self.detach_prev    = detach_prev
        self.ridge          = ridge
        self.ema            = ema
        self.pos_chunk_sz   = pos_chunk_sz
        self._initialized   = False
        self.pair_rng       = torch.Generator(device="cpu")
        self.pair_rng.manual_seed(int(seed) + 1009)

        self.readout = FrozenSoftcapReadout(
            raw_model.out_norm.weight, raw_model.tok_emb.weight, softcap=self.cfg.softcap
        ).to(device)

        if layer_mode == "random_gap4":
            if self.cfg.n_layers >= 12:
                self.valid_layer_pairs = [
                    (le, la) for le in range(2, 7) for la in range(6, 11) if (la - le) >= 4
                ]
            else:
                mg = max(1, self.cfg.n_layers // 3)
                self.valid_layer_pairs = [
                    (le, la) for le in range(1, self.cfg.n_layers - 1)
                    for la in range(2, self.cfg.n_layers) if (la - le) >= mg
                ]
            tracked_layers = sorted({l for pr in self.valid_layer_pairs for l in pr} | {early_layer, anchor_layer})
        elif layer_mode == "random_anc_7_10":
            if self.cfg.n_layers >= 12:
                anc_list = [7, 8, 9, 10]
            else:
                anc_list = list(range(max(2, early_layer + 1), self.cfg.n_layers))
            self.valid_layer_pairs = [(early_layer, la) for la in anc_list]
            tracked_layers = sorted(set(anc_list) | {early_layer, anchor_layer})
        else:
            self.valid_layer_pairs = [(early_layer, anchor_layer)]
            tracked_layers = sorted({early_layer, anchor_layer})

        self.tracked_layers = tracked_layers
        self.lenses = nn.ModuleDict({
            str(l): WorkspaceJLens(self.cfg.d_model, self.readout, device).to(device)
            for l in tracked_layers
        })
        self.lens_early  = self.lenses[str(early_layer)]
        self.lens_anchor = self.lenses[str(anchor_layer)]

        w_vec = torch.tensor([RHO_HARMONIC ** m for m in range(window)], dtype=torch.float32, device=device)
        self.win_weights = w_vec / w_vec.sum()

    def estimate_call_flops(
        self,
        micro_batch: int,
        n_act: int | None = None,
        n_uniq_seqs: int | None = None,
        early_layer: int | None = None,
        anchor_layer: int | None = None,
    ) -> float:
        """Exact FLOPs per `compute_loss` call accounting for shared per-sequence KV extraction + multi-position probes."""
        T   = self.cfg.seq_len
        D   = self.cfg.d_model
        F_d = self.cfg.d_ff
        V   = self.cfg.vocab_pad
        P   = self.k_dirs + 2
        W_a = self.window

        if n_act is None:
            if self.active_frac > 0.0:
                n_act = max(2, int(round(self.active_frac * micro_batch * T)))
                n_uniq_seqs = micro_batch
            else:
                n_act = min(self.probe_seqs, micro_batch)
                n_uniq_seqs = n_act
        if n_uniq_seqs is None:
            n_uniq_seqs = min(micro_batch, n_act)

        ear_l = self.early_layer if early_layer is None else early_layer
        anc_l = self.anchor_layer if anchor_layer is None else anchor_layer
        n_anc_blk = self.cfg.n_layers - anc_l
        n_ear_blk = (anc_l - ear_l) if self.enable_early else 0

        # 1) Shared clean KV extraction (computed ONCE per unique sequence via KV-only `wqkv.weight[D:]` slice!)
        kv_proj_cols = (2 * D) if self.cfg.linear_mode == "bf16" else (3 * D)
        kv_flops = (n_anc_blk + n_ear_blk) * (2.0 * n_uniq_seqs * T * D * kv_proj_cols)
        # 2) Probe block linear + Flash-Prefix causal attention (avg prefix length = 0.5 * T + W)
        blk_lin_2n = 2.0 * (D * 3 * D + D * D + D * 2 * F_d + F_d * D)
        anc_tok = n_act * P * W_a
        anc_flops = n_anc_blk * (anc_tok * blk_lin_2n + 4.0 * anc_tok * (0.5 * T + W_a) * D)
        ear_tok = n_act * P * 1
        ear_flops = n_ear_blk * (ear_tok * blk_lin_2n + 4.0 * ear_tok * (0.5 * T + 1) * D)
        # 3) Readout FLOPs (topk pool 2NV + selected fwd/bwd 4NV = 6NV per clean position + 2NV per probe token)
        ro_anc = 2.0 * (anc_tok * D * V) + 6.0 * (n_act * (D * D + D * V))
        ro_ear = (2.0 * (ear_tok * (D * D + D * V)) + 6.0 * (n_act * (D * D + D * V))) if self.enable_early else 0.0
        return kv_flops + anc_flops + ear_flops + ro_anc + ro_ear

    @torch.no_grad()
    def refresh_snapshot(self, raw_model: nn.Module, interm_det: list[Tensor], x_raw_final: Tensor):
        with torch.amp.autocast(device_type=x_raw_final.device.type, enabled=False):
            alpha = self.ema if self._initialized else 0.0
            self.readout.update_ema_(raw_model.out_norm.weight, raw_model.tok_emb.weight, ema=alpha)

            B, T, D = x_raw_final.shape
            M_tot   = B * T
            K_r     = min(4096, M_tot)
            ridx    = torch.randint(0, M_tot, (K_r,), device=x_raw_final.device)
            # Fix #3: Match Wave8 (036) — apply `pure_rms_norm_fp32` so 10x-norm sink tokens cannot hijack X^T X by 100x!
            y_fin   = pure_rms_norm_fp32(x_raw_final.reshape(M_tot, D)[ridx])
            inv_k   = 1.0 / float(K_r)
            eye     = torch.eye(D, dtype=torch.float32, device=x_raw_final.device).unsqueeze(0)

            # Batched Cholesky solve across all tracked layers (2 in `fixed`, 9 in `random_gap4`, <0.04ms/step amortized!)
            x_stk = torch.stack(
                [pure_rms_norm_fp32(interm_det[l_idx - 1].reshape(M_tot, D)[ridx]) for l_idx in self.tracked_layers],
                dim=0,
            )  # [L, K_r, D]
            g_xx = torch.matmul(x_stk.transpose(-1, -2), x_stk) * inv_k
            tr_g = g_xx.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).clamp_min(1e-6).unsqueeze(-1)
            cxy  = torch.einsum("lkd,ke->lde", x_stk, y_fin) * inv_k
            chol = torch.linalg.cholesky(g_xx + (self.ridge * tr_g) * eye)
            j_all = torch.cholesky_solve(cxy, chol).transpose(-1, -2).contiguous()

            x_c = x_stk - x_stk.mean(dim=1, keepdim=True)
            cov = torch.matmul(x_c.transpose(-1, -2), x_c) * inv_k
            tr_c = cov.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).clamp_min(1e-6).unsqueeze(-1)
            sig_all = cov / tr_c + 0.02 * eye

            for idx_l, l_idx in enumerate(self.tracked_layers):
                lens = self.lenses[str(l_idx)]
                lens.J.mul_(alpha).add_(j_all[idx_l], alpha=(1.0 - alpha))
                lens.Sigma.mul_(alpha).add_(sig_all[idx_l], alpha=(1.0 - alpha))

            self._initialized = True

    @torch.no_grad()
    def select_probe_positions(self, yb: Tensor, tok_nll_det: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Returns `(b_each [N_act], pos_each [N_act], uniq_b [B_u], kv_slot [N_act])`:
          - When `active_frac > 0`: samples `N_act = round(active_frac * B * T)` positions uniformly in a 2:1 ratio
            from the batch-wide Top-10% Hardest and Bottom-5% Easiest valid token pools, sorted by sequence `b` so
            all `M_seq` positions on sequence `b` share its single clean `(K, V)` cache!
          - When `active_frac <= 0`: legacy 1-position-per-sequence (`probe_seqs`) fallback.
        """
        B, T = yb.shape
        W    = self.window
        T_v  = T - W + 1
        M_tot = B * T

        valid_bt = torch.ones((B, T), dtype=torch.bool, device=yb.device)
        if W > 1:
            valid_bt[:, T_v:] = False
        for m in range(W):
            valid_bt[:, :T_v] &= (yb[:, m : T_v + m] != EOT_TOKEN)

        if self.active_frac > 0.0:
            valid_flat = valid_bt.reshape(-1)
            n_valid = int(valid_flat.sum().item())
            n_act_tot = min(n_valid, max(2, int(round(self.active_frac * M_tot))))

            if self.pos_mode == "uniform":
                val_idx = torch.nonzero(valid_flat, as_tuple=True)[0]
                sel_flat = val_idx[torch.randperm(val_idx.numel(), device=yb.device)[:n_act_tot]]
            else:
                nll_flat = tok_nll_det.reshape(-1)
                k_pool_hard = min(n_valid, max(2, int(math.ceil(max(0.10, (2.0 / 3.0) * self.active_frac) * M_tot))))
                nll_h = torch.where(valid_flat, nll_flat, torch.full_like(nll_flat, -1e9))
                _, hard_pool = torch.topk(nll_h, k_pool_hard, largest=True)

                is_hard = torch.zeros_like(valid_flat)
                is_hard[hard_pool] = True
                easy_elig = valid_flat & (~is_hard) & (nll_flat > 0.05)
                min_easy_pool = max(2, int(math.ceil(max(0.05, (1.0 / 3.0) * self.active_frac) * M_tot)))
                if int(easy_elig.sum().item()) < min_easy_pool:
                    easy_elig = valid_flat & (~is_hard)
                n_easy_elig = int(easy_elig.sum().item())
                k_pool_easy = min(max(1, n_easy_elig), min_easy_pool)
                nll_e = torch.where(easy_elig, nll_flat, torch.full_like(nll_flat, 1e9))
                _, easy_pool = torch.topk(nll_e, k_pool_easy, largest=False)

                n_take_hard = min(hard_pool.numel(), int(round((2.0 / 3.0) * n_act_tot)))
                n_take_easy = min(easy_pool.numel(), max(1, n_act_tot - n_take_hard))
                if n_take_hard + n_take_easy < n_act_tot and hard_pool.numel() >= n_act_tot - n_take_easy:
                    n_take_hard = n_act_tot - n_take_easy

                sel_h = hard_pool[torch.randperm(hard_pool.numel(), device=yb.device)[:n_take_hard]]
                sel_e = easy_pool[torch.randperm(easy_pool.numel(), device=yb.device)[:n_take_easy]]
                sel_flat = torch.cat([sel_h, sel_e], dim=0)

            sel_flat, _ = sel_flat.sort()
            b_each   = sel_flat // T
            pos_each = sel_flat % T
            uniq_b, kv_slot = torch.unique_consecutive(b_each, return_inverse=True)
            return b_each, pos_each, uniq_b, kv_slot

        # Legacy 1-position-per-sequence (`active_frac == 0.0`) mode:
        B_p = min(self.probe_seqs, B)
        valid_pos = valid_bt[:, :T_v]
        nll_bt = tok_nll_det.view(B, T)[:, :T_v]
        if self.pos_mode == "uniform":
            b_sel = torch.randperm(B, device=yb.device)[:B_p].sort()[0]
            rnd   = torch.rand((B_p, T_v), device=yb.device).masked_fill(~valid_pos[b_sel], -1.0)
            pos   = rnd.argmax(dim=-1)
            return b_sel, pos, b_sel, torch.arange(B_p, device=yb.device)

        win_nlls = torch.stack([tok_nll_det.view(B, T)[:, m : T_v + m] for m in range(W)], dim=-1)
        look_max = win_nlls.amax(dim=-1).masked_fill(~valid_pos, -1e9)
        seq_peak, best_hard_pos = look_max.max(dim=-1)
        n_hard = (2 * B_p) // 3
        n_easy = B_p - n_hard
        _, hard_seqs = torch.topk(seq_peak, n_hard, largest=True)
        rem_mask = torch.ones((B,), dtype=torch.bool, device=yb.device)
        rem_mask[hard_seqs] = False
        rem_seqs = torch.nonzero(rem_mask, as_tuple=True)[0]
        easy_seqs = rem_seqs[torch.randperm(rem_seqs.numel(), device=yb.device)[:n_easy]]
        easy_scores = torch.where(
            valid_pos[easy_seqs] & (nll_bt[easy_seqs] > 0.05),
            nll_bt[easy_seqs],
            torch.full_like(nll_bt[easy_seqs], 1e9),
        )
        best_easy_pos = easy_scores.argmin(dim=-1)
        b_raw = torch.cat([hard_seqs, easy_seqs], dim=0)
        p_raw = torch.cat([best_hard_pos[hard_seqs], best_easy_pos], dim=0)
        ord_b = b_raw.argsort()
        b_sel = b_raw[ord_b]
        pos   = p_raw[ord_b]
        uniq_b, kv_slot = torch.unique_consecutive(b_sel, return_inverse=True)
        return b_sel, pos, uniq_b, kv_slot

    @torch._dynamo.disable
    def compute_loss(
        self,
        raw_model: nn.Module,
        interm: list[Tensor],
        yb: Tensor,
        tok_nll_det: Tensor,
    ) -> tuple[Tensor, Tensor, dict]:
        W = self.window
        k = self.k_dirs
        P = k + 2
        B_tot, T_tot = yb.shape
        inv_m_tot = 1.0 / float(B_tot * T_tot)  # Fix #1: true per-token nat parity across all % active rungs!

        if self.layer_mode in ("random_gap4", "random_anc_7_10"):
            p_idx = int(torch.randint(0, len(self.valid_layer_pairs), (1,), generator=self.pair_rng).item())
            cur_ear, cur_anc = self.valid_layer_pairs[p_idx]
        else:
            cur_ear, cur_anc = self.early_layer, self.anchor_layer
        lens_ear = self.lenses[str(cur_ear)]
        lens_anc = self.lenses[str(cur_anc)]

        b_each, pos_each, uniq_b, kv_slot = self.select_probe_positions(yb, tok_nll_det)
        N_act = b_each.shape[0]
        B_u   = uniq_b.shape[0]
        c_sz  = self.pos_chunk_sz
        use_fast_gpu = yb.is_cuda and (self.cfg.head_dim == 64) and (self.cfg.d_model == 512)
        kv_fn  = extract_clean_kv_fast if use_fast_gpu else extract_clean_kv_for_blocks
        sup_fn = _compiled_suppress_oblique if use_fast_gpu else suppress_oblique_single_and_joint
        if use_fast_gpu:
            def adv_fn(blocks, kv_list, h_win, positions, cos_full, sin_full, kv_slot):
                wpe = 2 if h_win.shape[2] > 1 else 0
                return advance_local_window_kv_ultra(
                    blocks, kv_list, h_win, positions, cos_full, sin_full, kv_slot,
                    block_t=64, num_warps=4, waves_per_eu=wpe,
                )
        else:
            adv_fn = advance_local_window_kv

        # --- 1. Anchor Layer (`cur_anc .. n_layers`, window `W`) ---
        h_anc_seq = interm[cur_anc - 1]
        if self.detach_prev and cur_anc >= 2:
            prev_a = interm[cur_anc - 2].detach()
            h_anc_seq = prev_a + (h_anc_seq - prev_a)
        x_anc_live = h_anc_seq[b_each, pos_each]                # [N_act, D], grad-attached!

        pool_a = lens_anc.topk_pool_ids(x_anc_live, pool_size=self.pool_size)
        ids_a, w_a, d_a, val_a = select_diverse_from_pool(
            pool_a, lens_anc, k=k, max_cosine=self.max_cosine, metric=self.metric
        )

        with torch.no_grad():
            anc_blocks = raw_model.blocks[cur_anc :]
            anc_inputs = [interm[m - 1].detach() for m in range(cur_anc, self.cfg.n_layers)]
            kv_anc = kv_fn(anc_blocks, anc_inputs, uniq_b, raw_model.cos, raw_model.sin)

            w_offs = torch.arange(W, device=yb.device)
            u_anc_parts = []
            usable_a_parts = []
            h_anc_det = interm[cur_anc - 1].detach()

            for s in range(0, N_act, c_sz):
                e = min(s + c_sz, N_act)
                nc = e - s
                probes_c, usa_c, usa_all_c = sup_fn(
                    x_anc_live[s:e].detach(), w_a[s:e], d_a[s:e], val_a[s:e],
                    strength=self.strength, match_norm=self.match_norm,
                )
                win_idx_c = pos_each[s:e, None] + w_offs[None, :]
                h_clean_c = h_anc_det[b_each[s:e, None], win_idx_c]
                h_win_c   = h_clean_c.unsqueeze(1).expand(nc, P, W, self.cfg.d_model).clone()
                h_win_c[:, :, 0, :] = probes_c

                fin_c = adv_fn(
                    anc_blocks, kv_anc, h_win_c, pos_each[s:e], raw_model.cos, raw_model.sin, kv_slot=kv_slot[s:e]
                )
                y_win_c = yb[b_each[s:e, None], win_idx_c]
                flat_fin = fin_c.reshape(-1, self.cfg.d_model)
                flat_y   = y_win_c[:, None, :].expand(nc, P, W).reshape(-1)
                nll_pw   = self.readout.fast_nograd_target_nll(flat_fin, flat_y).view(nc, P, W)

                valid_y = (y_win_c != EOT_TOKEN).float()
                wt_y = valid_y * self.win_weights.unsqueeze(0)
                wt_y = wt_y / wt_y.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                scores_c = (nll_pw * wt_y.unsqueeze(1)).sum(dim=-1)

                raw_u_c = scores_c[:, 1 : k + 1] - scores_c[:, 0:1]
                u_all_c = scores_c[:, k + 1] - scores_c[:, 0]
                u_anc_parts.append(calibrate_usefulness(raw_u_c, u_all_c, usa_c, usa_all_c, self.coalition_norm))
                usable_a_parts.append(usa_c)

            u_anc    = torch.cat(u_anc_parts, dim=0)
            usable_a = torch.cat(usable_a_parts, dim=0)
            del kv_anc

        lq_sel_anc     = lens_anc.selected_log_probs(x_anc_live, ids_a)
        vec_anc        = compute_feature_loss_pregathered(
            lq_sel_anc, u_anc, usable_a, loss_kind=self.loss_kind, exp_beta=self.exp_beta, exp_norm=self.exp_norm
        )
        loss_anc_mean  = vec_anc.mean()
        loss_anc_batch = vec_anc.sum() * inv_m_tot

        # --- 2. Early-Layer Telescoped Probes (`cur_ear .. cur_anc`, `W_early = 1` at `pos_each`!) ---
        loss_ear_mean  = torch.zeros((), dtype=torch.float32, device=yb.device)
        loss_ear_batch = torch.zeros((), dtype=torch.float32, device=yb.device)
        u_ear_abs = 0.0
        if self.enable_early:
            h_ear_seq = interm[cur_ear - 1]
            if self.detach_prev and cur_ear >= 2:
                prev_e = interm[cur_ear - 2].detach()
                h_ear_seq = prev_e + (h_ear_seq - prev_e)
            x_ear_live = h_ear_seq[b_each, pos_each]

            pool_e = lens_ear.topk_pool_ids(x_ear_live, pool_size=self.pool_size)
            ids_e, w_e, d_e, val_e = select_diverse_from_pool(
                pool_e, lens_ear, k=k, max_cosine=self.max_cosine, metric=self.metric
            )

            with torch.no_grad():
                ear_blocks = raw_model.blocks[cur_ear : cur_anc]
                ear_inputs = [interm[m - 1].detach() for m in range(cur_ear, cur_anc)]
                kv_ear = kv_fn(ear_blocks, ear_inputs, uniq_b, raw_model.cos, raw_model.sin)

                if self.hop_norm:
                    norm_a = u_anc.abs().sum(dim=-1, keepdim=True).clamp_min(0.05)
                    u_anc_target = u_anc / norm_a
                else:
                    u_anc_target = u_anc

                u_ear_parts = []
                usable_e_parts = []
                for s in range(0, N_act, c_sz):
                    e = min(s + c_sz, N_act)
                    nc = e - s
                    probes_ec, usa_ec, usa_all_ec = sup_fn(
                        x_ear_live[s:e].detach(), w_e[s:e], d_e[s:e], val_e[s:e],
                        strength=self.strength, match_norm=self.match_norm,
                    )
                    at_anc_c = adv_fn(
                        ear_blocks, kv_ear, probes_ec.unsqueeze(2), pos_each[s:e],
                        raw_model.cos, raw_model.sin, kv_slot=kv_slot[s:e],
                    ).squeeze(2)

                    with torch.amp.autocast(device_type=yb.device.type, enabled=False):
                        at_anc_h = lens_anc.project_hidden(at_anc_c.reshape(nc * P, self.cfg.d_model))
                    ids_ac = ids_a[s:e].unsqueeze(1).expand(nc, P, k).reshape(nc * P, k)
                    ut_c   = u_anc_target[s:e].unsqueeze(1).expand(nc, P, k).reshape(nc * P, k)
                    usa_ac = usable_a[s:e].unsqueeze(1).expand(nc, P, k).reshape(nc * P, k)
                    lq_ec  = self.readout.fast_nograd_selected_logprobs(at_anc_h, ids_ac)
                    sc_ec  = compute_feature_loss_pregathered(
                        lq_ec, ut_c, usa_ac, loss_kind=self.loss_kind, exp_beta=self.exp_beta, exp_norm=self.exp_norm
                    ).view(nc, P)

                    raw_u_ec = sc_ec[:, 1 : k + 1] - sc_ec[:, 0:1]
                    u_all_ec = sc_ec[:, k + 1] - sc_ec[:, 0]
                    u_ear_parts.append(calibrate_usefulness(raw_u_ec, u_all_ec, usa_ec, usa_all_ec, self.coalition_norm))
                    usable_e_parts.append(usa_ec)

                u_ear    = torch.cat(u_ear_parts, dim=0)
                usable_e = torch.cat(usable_e_parts, dim=0)
                u_ear_abs = float(u_ear.abs().mean().item())
                del kv_ear

            lq_sel_ear     = lens_ear.selected_log_probs(x_ear_live, ids_e)
            vec_ear        = compute_feature_loss_pregathered(
                lq_sel_ear, u_ear, usable_e, loss_kind=self.loss_kind, exp_beta=self.exp_beta, exp_norm=self.exp_norm
            )
            loss_ear_mean  = vec_ear.mean()
            loss_ear_batch = vec_ear.sum() * inv_m_tot

        tot_aux  = 0.5 * (loss_anc_batch + loss_ear_batch) if self.enable_early else loss_anc_batch
        tot_obs  = (0.5 * (loss_anc_mean + loss_ear_mean) if self.enable_early else loss_anc_mean).detach()
        stats = {
            "l_ear": int(cur_ear),
            "l_anc": int(cur_anc),
            "n_act": int(N_act),
            "n_seqs": int(B_u),
            "u_anc_abs": round(float(u_anc.abs().mean().item()), 5),
            "u_anc_pos_frac": round(float((u_anc > 0).float().mean().item()), 3),
            "u_ear_abs": round(u_ear_abs, 5),
            "loss_anc": round(float(loss_anc_mean.detach().item()), 5),
            "loss_ear": round(float(loss_ear_mean.detach().item()), 5),
            "j_batch": round(float(tot_aux.detach().item()), 5),
        }
        return tot_aux, tot_obs, stats
