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
from pathlib import Path
import numpy as np
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
from ultra_probe_kernels import advance_local_window_kv_ultra, _softcap_selected_bwd_2half_kernel

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
    Saves only `hn` [N, D] in BF16 + `lse` [N] in FP32, running BF16 MFMA GEMMs in chunks with exact FP32
    selected-slot gradient subtraction (preventing BF16 catastrophic cancellation at high `p >= 0.99`).
    """
    @staticmethod
    def forward(ctx, hn_fp32: Tensor, U_bf16: Tensor, ids: Tensor, softcap: float, chunk_sz: int = 4096) -> Tensor:
        N, D = hn_fp32.shape
        k = ids.shape[1]
        V = U_bf16.shape[0]
        inv_sc = (1.0 / softcap) if softcap > 0.0 else 0.0
        use_2half = (32768 < V <= 65536)

        hn_bf_all = hn_fp32.detach().to(torch.bfloat16)
        lq_sel = torch.empty((N, k), dtype=torch.float32, device=hn_fp32.device)
        lse_all = torch.empty((N,), dtype=torch.float32, device=hn_fp32.device)
        buf = torch.empty((min(N, chunk_sz), V), dtype=torch.bfloat16, device=hn_fp32.device)

        for s in range(0, N, chunk_sz):
            e = min(s + chunk_sz, N)
            nr = e - s
            hn_bf = hn_bf_all[s:e]
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
            lg_k = (softcap * torch.tanh((raw_k * inv_sc).clamp(-20.0, 20.0))) if softcap > 0.0 else raw_k
            lq_sel[s:e] = lg_k - l_sl.unsqueeze(1)

        ctx.save_for_backward(hn_bf_all, U_bf16, ids, lse_all)
        ctx.hn_dtype = hn_fp32.dtype
        ctx.softcap = float(softcap)
        ctx.chunk_sz = int(chunk_sz)
        return lq_sel

    @staticmethod
    def backward(ctx, grad_lq: Tensor):
        hn_bf_saved, U_bf16, ids, lse_all = ctx.saved_tensors
        softcap = ctx.softcap
        chunk_sz = ctx.chunk_sz
        inv_sc = (1.0 / softcap) if softcap > 0.0 else 0.0

        N, D = hn_bf_saved.shape
        V = U_bf16.shape[0]
        use_2half = (32768 < V <= 65536)
        dhn_bf = torch.empty((N, D), dtype=torch.bfloat16, device=hn_bf_saved.device)
        buf = torch.empty((min(N, chunk_sz), V), dtype=torch.bfloat16, device=hn_bf_saved.device)

        for s in range(0, N, chunk_sz):
            e = min(s + chunk_sz, N)
            nr = e - s
            hn_bf = hn_bf_saved[s:e]
            b_sl = buf[:nr]
            l_sl = lse_all[s:e]
            torch.matmul(hn_bf, U_bf16.t(), out=b_sl)

            ids_sl = ids[s:e]
            raw_k = b_sl.gather(1, ids_sl).float()
            if softcap > 0.0:
                th_k = torch.tanh((raw_k * inv_sc).clamp(-20.0, 20.0))
                lg_k = softcap * th_k
                ds_k = 1.0 - th_k * th_k
            else:
                lg_k = raw_k
                ds_k = torch.ones_like(raw_k)

            g_k = grad_lq[s:e].float()
            sum_g = g_k.sum(dim=-1).contiguous()
            prob_k = torch.exp(lg_k - l_sl.unsqueeze(1))
            # Duplicate-safe FP32 sum of g_k at each selected vocab ID, followed by exact FP32 subtraction:
            eq_ids = (ids_sl.unsqueeze(2) == ids_sl.unsqueeze(1)).to(g_k.dtype)
            g_sum_k = torch.bmm(eq_ids, g_k.unsqueeze(2)).squeeze(2)
            sel_grad_fp32 = (g_sum_k - sum_g.unsqueeze(1) * prob_k) * ds_k

            if use_2half:
                _softcap_selected_bwd_2half_kernel[(nr,)](
                    b_sl, b_sl.stride(0), l_sl, sum_g,
                    V=V, SOFTCAP=softcap, INV_SOFTCAP=inv_sc, HALF_V=32768, num_warps=8,
                )
            else:
                _softcap_selected_base_bwd_kernel[(nr,)](
                    b_sl, b_sl.stride(0), l_sl, sum_g,
                    V=V, SOFTCAP=softcap, INV_SOFTCAP=inv_sc, BLOCK_V=4096, num_warps=8,
                )
            # Overwrite selected slots with exact FP32-subtracted BF16 gradients (identical across any duplicate IDs):
            b_sl.scatter_(1, ids_sl, sel_grad_fp32.to(torch.bfloat16))
            torch.matmul(b_sl, U_bf16, out=dhn_bf[s:e])

        return dhn_bf.to(ctx.hn_dtype), None, None, None, None


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
    def fast_topk_pool_ids_and_probs(self, h_2d: Tensor, pool_size: int = 64, chunk_sz: int = 16384) -> tuple[Tensor, Tensor]:
        """Finds top-`pool_size` token IDs `[N, M]` AND softcapped pool probs `[N, M]` with ZERO extra GEMMs."""
        M_pool = min(pool_size, self.U_bf16.shape[0])
        if not h_2d.is_cuda or self.U_bf16.shape[0] < 1024:
            top_vals, top_ids = self.logits(h_2d).topk(M_pool, dim=-1)
            return top_ids, F.softmax(top_vals.float(), dim=-1)
        N, D = h_2d.shape
        V = self.U_bf16.shape[0]
        from fast_kernels import FusedRMSNormFn
        hn_all = FusedRMSNormFn.apply(h_2d.to(torch.bfloat16), self.norm_w, self.eps) if D == 512 else None
        out_ids   = torch.empty((N, M_pool), dtype=torch.long, device=h_2d.device)
        out_probs = torch.empty((N, M_pool), dtype=torch.float32, device=h_2d.device)
        buf = torch.empty((min(N, chunk_sz), V), dtype=torch.bfloat16, device=h_2d.device)
        inv_sc = (1.0 / self.softcap) if self.softcap > 0.0 else 0.0
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
            top_vals, top_ids = b_sl.topk(M_pool, dim=-1)
            tv = top_vals.float()
            if self.softcap > 0.0:
                tv = self.softcap * torch.tanh(tv * inv_sc)
            out_ids[s:e]   = top_ids
            out_probs[s:e] = F.softmax(tv, dim=-1)
        return out_ids, out_probs

    @torch.no_grad()
    def fast_topk_pool_ids(self, h_2d: Tensor, pool_size: int = 64, chunk_sz: int = 16384) -> Tensor:
        """Finds top-`pool_size` token IDs `[N, M]` without allocating `[N, V]` FP32 tensors."""
        top_ids, _ = self.fast_topk_pool_ids_and_probs(h_2d, pool_size=pool_size, chunk_sz=chunk_sz)
        return top_ids

    def selected_log_probs(self, h_2d: Tensor, ids_2d: Tensor) -> Tensor:
        """Differentiable `log_probs(h_2d).gather(-1, ids_2d)` `[N, k]` with O(N*D) saved memory."""
        if not h_2d.is_cuda or self.U_bf16.shape[0] < 1024:
            return self.log_probs(h_2d).gather(-1, ids_2d)
        with torch.amp.autocast(device_type=h_2d.device.type, enabled=False):
            hf = h_2d.float()
            hn = hf * torch.rsqrt(hf.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.norm_w
            return _SelectedSoftcapLinearLogProbsFn.apply(hn, self.U_bf16, ids_2d, self.softcap, 16384)

    @torch.no_grad()
    def fast_nograd_target_nll(
        self,
        h_2d: Tensor,
        targets_1d: Tensor,
        chunk_sz: int = 32768,
        norm_w_override: Tensor | None = None,
        U_bf16_override: Tensor | None = None,
        U_fp32_override: Tensor | None = None,
    ) -> Tensor:
        nw   = self.norm_w if norm_w_override is None else norm_w_override
        u_bf = self.U_bf16 if U_bf16_override is None else U_bf16_override
        if not h_2d.is_cuda or u_bf.shape[0] < 1024:
            if norm_w_override is None and U_fp32_override is None:
                return F.cross_entropy(self.logits(h_2d), targets_1d, reduction="none")
            u_fp = self.U if U_fp32_override is None else U_fp32_override
            with torch.amp.autocast(device_type=h_2d.device.type, enabled=False):
                hf  = h_2d.float()
                hn  = hf * torch.rsqrt(hf.pow(2).mean(dim=-1, keepdim=True) + self.eps) * nw.float()
                raw = F.linear(hn, u_fp.float())
                lg  = (self.softcap * torch.tanh(raw * (1.0 / self.softcap))) if self.softcap > 0.0 else raw
                return F.cross_entropy(lg, targets_1d, reduction="none")
        N, D = h_2d.shape
        V = u_bf.shape[0]
        inv_sc = (1.0 / self.softcap) if self.softcap > 0.0 else 0.0
        use_2half = (32768 < V <= 65536)
        from fast_kernels import FusedRMSNormFn
        hn_all = FusedRMSNormFn.apply(h_2d.to(torch.bfloat16), nw, self.eps) if D == 512 else None
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
                hn = (hf * torch.rsqrt(hf.pow(2).mean(dim=-1, keepdim=True) + self.eps) * nw).to(torch.bfloat16)
            b_sl = buf[:nr]
            torch.matmul(hn, u_bf.t(), out=b_sl)
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
    """Layer lens `J_l` (`ols_norm`, `identity`, or exact `true_jacobian` `E[dx_L/dx_l]`) + shared detached readout."""

    def __init__(
        self,
        d_model: int,
        readout: FrozenSoftcapReadout,
        device: torch.device,
        norm_input: bool = True,
        is_identity: bool = False,
        norm_w_dir: bool = False,
    ):
        super().__init__()
        self.d_model     = d_model
        self.readout     = readout
        self.norm_input  = bool(norm_input)
        self.is_identity = bool(is_identity)
        self.norm_w_dir  = bool(norm_w_dir)
        self.register_buffer("J", torch.eye(d_model, dtype=torch.float32, device=device))
        self.register_buffer("Sigma", torch.eye(d_model, dtype=torch.float32, device=device))

    def project_hidden(self, x: Tensor) -> Tensor:
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            xf = pure_rms_norm_fp32(x) if self.norm_input else x.float()
            if self.is_identity:
                return xf
            return xf @ self.J.t()

    def log_probs(self, x: Tensor) -> Tensor:
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            return self.readout.log_probs(self.project_hidden(x))

    @torch.no_grad()
    def topk_pool_ids_and_probs(self, x: Tensor, pool_size: int = 64) -> tuple[Tensor, Tensor]:
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            return self.readout.fast_topk_pool_ids_and_probs(self.project_hidden(x.detach()), pool_size=pool_size)

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
            if self.norm_w_dir:
                u_tok = u_tok * self.readout.norm_w
            w_read = u_tok if self.is_identity else torch.matmul(u_tok, self.J)
            if metric == "cov":
                d_write = torch.matmul(w_read, self.Sigma)
                denom = (w_read * d_write).sum(dim=-1, keepdim=True).clamp_min(1e-16).sqrt()
                u_norm = u_tok.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                u_sim  = u_tok / u_norm
                u_sim  = torch.where(denom > 1e-8, u_sim, torch.zeros_like(u_sim))
            else:
                d_write = w_read
                w_len   = w_read.norm(dim=-1, keepdim=True)
                u_sim   = torch.where(w_len > 1e-8, w_read / w_len.clamp_min(1e-8), torch.zeros_like(w_read))
            return w_read, d_write, u_sim


def _cluster_select_indices(
    G: Tensor,
    pool_probs: Tensor,
    valid_cand: Tensor,
    k: int,
    select_mode: str,
) -> tuple[Tensor, Tensor]:
    """
    Threshold-free candidate selection over `[B, M_pool]` Gram matrix `G = u_sim @ u_sim.T`:
      - `maxmin`: Furthest-point / k-center non-similar seeded at rank-0 (`argmax_j p_j^0.15 * (1 - max_sim)`).
      - `kmeans`: 3-step Gram-kernelized Spherical k-Means (`sqrt(p)`-weighted signed assignments) + distinct medoids.
      - `emeans`: 3-step Gram-kernelized vMF Soft-EM (`tau = clamp(std(|S|), min=0.05)`) + distinct medoids.
    """
    B, M_pool, _ = G.shape
    k_eff = min(k, M_pool)
    b_ar  = torch.arange(B, device=G.device)
    G_abs = G.abs()
    p_f   = pool_probs.float().clamp_min(1e-12)
    p_w   = p_f.pow(0.15)

    chosen  = torch.zeros((B, M_pool), dtype=torch.bool, device=G.device)
    sel_m   = torch.zeros((B, k), dtype=torch.long, device=G.device)
    val_out = torch.zeros((B, k), dtype=torch.bool, device=G.device)

    # 1. `maxmin` furthest-point seed (slot 0 = rank-0 when valid)
    has_0 = valid_cand.any(dim=-1)
    c0    = torch.where(valid_cand[:, 0], torch.zeros_like(b_ar), valid_cand.int().argmax(dim=-1))
    sel_m[:, 0]   = c0
    val_out[:, 0] = has_0
    chosen[b_ar, c0] = has_0
    max_sim = torch.where(has_0[:, None], G_abs[b_ar, :, c0], torch.zeros((B, M_pool), dtype=torch.float32, device=G.device))

    for r in range(1, k_eff):
        avail = valid_cand & (~chosen)
        has_r = avail.any(dim=-1)
        score = (p_w * (1.0 - max_sim).clamp_min(0.0) + 1e-7 * p_w).masked_fill(~avail, -1e9)
        cr    = score.argmax(dim=-1)
        sel_m[:, r]   = cr
        val_out[:, r] = has_r
        chosen[b_ar, cr] = chosen[b_ar, cr] | has_r
        max_sim = torch.where(has_r[:, None], torch.maximum(max_sim, G_abs[b_ar, :, cr]), max_sim)

    if select_mode == "maxmin":
        return sel_m, val_out

    # 2. Gram-kernelized Spherical `kmeans` / vMF `emeans` (3 iterations in `[B, M_pool, k]`)
    A = torch.zeros((B, M_pool, k), dtype=torch.float32, device=G.device)
    A.scatter_(1, sel_m.unsqueeze(1), val_out.float().unsqueeze(1))
    sqrt_p = pool_probs.float().clamp_min(0.0).sqrt() * valid_cand.float()

    for _ in range(3):
        S     = torch.bmm(G, A)
        S_abs = S.abs()
        sgn   = torch.where(S >= 0.0, torch.ones_like(S), -torch.ones_like(S))
        if select_mode == "kmeans":
            r_star = S_abs.masked_fill(~val_out.unsqueeze(1), -1e9).argmax(dim=-1, keepdim=True)
            W_assign = torch.zeros_like(S).scatter_(2, r_star, 1.0)
            W = W_assign * sqrt_p.unsqueeze(-1) * sgn
        else:  # `emeans`
            tau = (0.15 * S_abs.std(dim=(-2, -1), keepdim=True, unbiased=False)).clamp(0.02, 0.08)
            gamma = F.softmax((S_abs / tau).masked_fill(~val_out.unsqueeze(1), -1e9), dim=-1)
            W = sqrt_p.unsqueeze(-1) * gamma * sgn

        GW      = torch.bmm(G, W)
        norm_sq = (W * GW).sum(dim=1, keepdim=True)
        A_new   = W * torch.rsqrt(norm_sq.clamp_min(1e-12))
        A       = torch.where(norm_sq > 1e-10, A_new, A)

    # 3. Extract 1 distinct medoid token per cluster `r = 0..k-1`
    S_fin  = torch.bmm(G, A).abs()
    r_best = S_fin.masked_fill(~val_out.unsqueeze(1), -1e9).argmax(dim=-1)

    med_m   = torch.zeros((B, k), dtype=torch.long, device=G.device)
    med_val = torch.zeros((B, k), dtype=torch.bool, device=G.device)
    used    = torch.zeros((B, M_pool), dtype=torch.bool, device=G.device)
    cur_max_sim = torch.zeros((B, M_pool), dtype=torch.float32, device=G.device)

    for r in range(k_eff):
        avail     = valid_cand & (~used)
        has_avail = avail.any(dim=-1)
        in_clust  = avail & (r_best == r)
        has_clust = in_clust.any(dim=-1)

        c_med = (p_f * S_fin[:, :, r]).masked_fill(~in_clust, -1e9).argmax(dim=-1)
        if r == 0:
            c_fb = p_f.masked_fill(~avail, -1e9).argmax(dim=-1)
        else:
            c_fb = (p_w * (1.0 - cur_max_sim).clamp_min(0.0) + 1e-7 * p_w).masked_fill(~avail, -1e9).argmax(dim=-1)

        cr = torch.where(has_clust, c_med, c_fb)
        med_m[:, r]   = cr
        med_val[:, r] = has_avail
        used[b_ar, cr] = used[b_ar, cr] | has_avail
        cur_max_sim = torch.where(has_avail[:, None], torch.maximum(cur_max_sim, G_abs[b_ar, :, cr]), cur_max_sim)

    return med_m, med_val


_compiled_cluster_select_indices = torch.compile(_cluster_select_indices, dynamic=True)


@torch.no_grad()
def select_diverse_from_pool(
    pool: Tensor,
    lens: WorkspaceJLens,
    k: int = 8,
    max_cosine: float = 0.85,
    metric: str = "cov",
    select_mode: str = "greedy_cos",
    pool_probs: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    assert select_mode in ("greedy_cos", "maxmin", "kmeans", "emeans"), f"Unknown select_mode={select_mode}"
    M_pool = pool.shape[-1]
    with torch.amp.autocast(device_type=pool.device.type, enabled=False):
        w_pool, d_pool, u_sim = lens.candidate_vectors(pool, metric=metric)
        B, _, D = w_pool.shape
        u_f = u_sim.float()
        sim_norms = u_f.norm(dim=-1)
        G_signed  = torch.matmul(u_f, u_f.transpose(-1, -2)).clamp(-1.0, 1.0)

        if select_mode != "greedy_cos":
            valid_cand = sim_norms > 0.5
            if pool_probs is None:
                # Fallback monotonically decaying rank prior if probs were not passed:
                rk = torch.arange(M_pool, device=pool.device, dtype=torch.float32)
                pool_probs = F.softmax(-0.15 * rk, dim=-1).unsqueeze(0).expand(B, M_pool)
            sel_fn = _compiled_cluster_select_indices if (pool.is_cuda and B >= 64) else _cluster_select_indices
            sel_m, final_val = sel_fn(G_signed, pool_probs, valid_cand, k, select_mode)
            ids_out = pool.gather(1, sel_m)
            m_exp   = sel_m.unsqueeze(-1).expand(B, k, D)
            w_out   = w_pool.float().gather(1, m_exp)
            d_out   = d_pool.float().gather(1, m_exp)
            ids_out = torch.where(final_val, ids_out, torch.zeros_like(ids_out))
            w_out   = torch.where(final_val.unsqueeze(-1), w_out, torch.zeros_like(w_out))
            d_out   = torch.where(final_val.unsqueeze(-1), d_out, torch.zeros_like(d_out))
            return ids_out, w_out, d_out, final_val

        cos_mat = G_signed.abs()
        if pool.is_cuda and M_pool <= 256:
            block_m = max(64, triton.next_power_of_2(M_pool))
            nw = 4 if block_m >= 128 else 2
            acc_i8 = torch.empty((B, M_pool), dtype=torch.int8, device=pool.device)
            _greedy_diverse_accept_kernel[(B,)](
                sim_norms.contiguous(), cos_mat.contiguous(), acc_i8,
                M_pool, k, float(max_cosine), BLOCK_M=block_m, num_warps=nw,
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
    select_mode: str = "greedy_cos",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    M_pool = min(pool_size, logq.shape[-1])
    top_vals, pool = logq.float().topk(M_pool, dim=-1)
    pool_probs = F.softmax(top_vals, dim=-1)
    return select_diverse_from_pool(
        pool, lens, k=k, max_cosine=max_cosine, metric=metric, select_mode=select_mode, pool_probs=pool_probs
    )


@torch.no_grad()
def suppress_oblique_single_and_joint(
    x: Tensor,
    w_read: Tensor,
    d_write: Tensor,
    valid: Tensor,
    strength: float = 1.0,
    match_norm: bool = True,
    include_clean: bool = True,
    include_joint: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        B, k, D = w_read.shape
        xf = x.float()
        w_read = w_read.float()
        d_write = d_write.float()
        old_norm = xf.norm(dim=-1, keepdim=True)
        n_extra  = (1 if include_clean else 0) + (1 if include_joint else 0)

        if strength <= 0.0:
            clean_rep = x.unsqueeze(1).expand(B, k + n_extra, D).contiguous()
            usa_all0  = valid.any(dim=-1) if include_joint else torch.zeros((B,), dtype=torch.bool, device=x.device)
            return clean_rep, valid.clone(), usa_all0

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

        parts = [xf.unsqueeze(1)] if include_clean else []
        parts.append(ed_k)

        if include_joint:
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
            parts.append(ed_all.unsqueeze(1))
        else:
            usable_all = torch.zeros((B,), dtype=torch.bool, device=x.device)

        probes = torch.cat(parts, dim=1).to(x.dtype)
        return probes, usable_k, usable_all

_compiled_suppress_oblique = torch.compile(suppress_oblique_single_and_joint, dynamic=True)


def _sampled_softcap_nll_vec(
    h_2d: Tensor,
    w_y: Tensor,
    w_cand: Tensor,
    y_1d: Tensor,
    cand_ids: Tensor,
    log_kc: Tensor,
    softcap: float,
) -> Tensor:
    """Importance-sampled Softcap-CE returning per-row NLL `[M]` (`[N_act * H_w]`) for shared-position multi-horizon lookahead."""
    raw_true = (h_2d.float() * w_y.float()).sum(dim=-1)
    raw_cand = torch.matmul(h_2d, w_cand.t()).float()
    if softcap > 0.0:
        inv_sc  = 1.0 / softcap
        lg_true = softcap * torch.tanh(raw_true * inv_sc)
        lg_cand = softcap * torch.tanh(raw_cand * inv_sc)
    else:
        lg_true, lg_cand = raw_true, raw_cand
    is_same  = (cand_ids.unsqueeze(0) == y_1d.unsqueeze(1))
    adj_cand = torch.where(is_same, torch.full_like(lg_cand, -1e9), lg_cand - log_kc.unsqueeze(0))
    return torch.logaddexp(lg_true, torch.logsumexp(adj_cand, dim=-1)) - lg_true


_compiled_sampled_softcap_nll_vec = torch.compile(_sampled_softcap_nll_vec, dynamic=True)


def compute_effective_feature_weights(
    weights: Tensor,
    valid: Tensor,
    exp_beta: float = 0.0,
    exp_norm: bool = True,
    bystander_tau: float = 0.0,
    bystander_w: float = 0.0,
    bystander_unscaled: bool = False,
    u_clamp: float = 0.6,
    shift_neg_by_bys: bool = True,
) -> Tensor:
    u_lim = float(u_clamp)
    u_c   = torch.where(valid, weights.detach().float().clamp(-u_lim, u_lim), torch.zeros_like(weights, dtype=torch.float32))
    tau   = max(0.0, float(bystander_tau))
    w_bys = max(0.0, float(bystander_w))

    # C^1-smooth exponential gain with linear continuation past z_max=3.6 (exact knee at u=0.6 for beta=6.0):
    # strictly monotonic up to u_clamp=5.0 without e^30 winner-take-all batch collapse under exp_norm=True!
    def _exp_gain(u_abs_in: Tensor) -> Tensor:
        if exp_beta <= 0.0:
            return u_abs_in
        z = exp_beta * u_abs_in
        z_c = z.clamp_max(3.6)
        return (torch.expm1(z_c) + math.exp(3.6) * (z - z_c)) * (1.0 / exp_beta)

    if tau > 0.0 or w_bys > 0.0:
        u_abs  = u_c.abs()
        phi    = _exp_gain(u_abs)
        is_pos = valid & (u_c >= tau) & (u_c > 0.0)
        is_neg = valid & (u_c <= -tau) & (u_c < 0.0)
        is_bys = valid & (~is_pos) & (~is_neg)

        if bystander_unscaled:
            # Normalize ONLY active (`|u_c| >= tau`) exponential terms; keep `|u_c| < tau` bystanders at full unshrunk `-w_bys`!
            if exp_norm and exp_beta > 0.0:
                is_act  = is_pos | is_neg
                n_act   = is_act.float().sum().clamp_min(1.0)
                mean_au = (u_abs * is_act.float()).sum() / n_act
                mean_ap = (phi * is_act.float()).sum() / n_act
                phi_eff = torch.where(mean_ap > 1e-8, phi * (mean_au / mean_ap.clamp_min(1e-8)), phi)
            else:
                phi_eff = phi
            neg_eff = (phi_eff + w_bys) if shift_neg_by_bys else phi_eff
            return torch.where(
                is_pos,
                phi_eff,
                torch.where(is_neg, -neg_eff, torch.where(is_bys, torch.full_like(u_c, -w_bys), torch.zeros_like(u_c))),
            )

        neg_raw = (phi + w_bys) if shift_neg_by_bys else phi
        u_raw = torch.where(
            is_pos,
            phi,
            torch.where(is_neg, -neg_raw, torch.where(is_bys, torch.full_like(u_c, -w_bys), torch.zeros_like(u_c))),
        )
        if exp_norm:
            nat_neg = (u_abs + w_bys) if shift_neg_by_bys else u_abs
            nat_ref = torch.where(
                is_pos,
                u_abs,
                torch.where(is_neg, nat_neg, torch.where(is_bys, torch.full_like(u_c, w_bys), torch.zeros_like(u_c))),
            )
            n_val  = valid.float().sum().clamp_min(1.0)
            mean_t = nat_ref.sum() / n_val
            mean_r = u_raw.abs().sum() / n_val
            return torch.where(mean_r > 1e-8, u_raw * (mean_t / mean_r.clamp_min(1e-8)), u_raw)
        return u_raw

    if exp_beta > 0.0:
        u_raw = torch.sign(u_c) * _exp_gain(u_c.abs())
        u_raw = torch.where(valid, u_raw, torch.zeros_like(u_raw))
        if exp_norm:
            n_val  = valid.float().sum().clamp_min(1.0)
            mean_c = u_c.abs().sum() / n_val
            mean_r = u_raw.abs().sum() / n_val
            return u_raw * (mean_c / mean_r.clamp_min(1e-8))
        return u_raw

    return u_c


def compute_feature_loss_pregathered(
    lq_sel: Tensor,
    weights: Tensor,
    valid: Tensor,
    loss_kind: str = "superpos_unlikelihood",
    exp_beta: float = 0.0,
    exp_norm: bool = True,
    bystander_tau: float = 0.0,
    bystander_w: float = 0.0,
    bystander_unscaled: bool = False,
) -> Tensor:
    with torch.amp.autocast(device_type=lq_sel.device.type, enabled=False):
        w = torch.where(valid, weights.detach().float(), torch.zeros_like(weights, dtype=torch.float32))
        lq_f = lq_sel.float()

        if loss_kind == "signed_raw":
            return -(w * lq_f).sum(dim=-1)

        if loss_kind == "exp_unlikelihood":
            u_tilde = compute_effective_feature_weights(
                w, valid, exp_beta=exp_beta, exp_norm=exp_norm,
                bystander_tau=bystander_tau, bystander_w=bystander_w,
                bystander_unscaled=bystander_unscaled, u_clamp=5.0, shift_neg_by_bys=False,
            )
            w_pos = F.relu(u_tilde)
            w_neg = F.relu(-u_tilde)
            q_sel = lq_f.exp().clamp(0.0, 1.0 - 1e-7)
            pos_mask = (u_tilde > 0.0) & valid
            sum_q_pos = (q_sel * pos_mask.float()).sum(dim=-1, keepdim=True)
            q_bg = (1.0 - sum_q_pos).clamp_min(1e-6)
            log_q_bg  = torch.log(q_bg)
            log_p_ind = lq_f - torch.logaddexp(lq_f, log_q_bg.expand_as(lq_f))
            pos_term  = -(w_pos * log_p_ind).sum(dim=-1)
            # Bounded probability-proportional unlikelihood: 0% gradient sign inversion, loss >= 0!
            neg_term  = -(w_neg * torch.log1p(-q_sel)).sum(dim=-1)
            return pos_term + neg_term

        if loss_kind == "exp_signed_clamped":
            u_tilde = compute_effective_feature_weights(
                w, valid, exp_beta=exp_beta, exp_norm=exp_norm,
                bystander_tau=bystander_tau, bystander_w=bystander_w, bystander_unscaled=bystander_unscaled,
            )
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
    bystander_tau: float = 0.0,
    bystander_w: float = 0.0,
    bystander_unscaled: bool = False,
) -> Tensor:
    lq_sel = logq.float().gather(-1, ids)
    return compute_feature_loss_pregathered(
        lq_sel, weights, valid, loss_kind=loss_kind, exp_beta=exp_beta,
        exp_norm=exp_norm, bystander_tau=bystander_tau, bystander_w=bystander_w,
        bystander_unscaled=bystander_unscaled,
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
    gamma_max: float = 4.0,
) -> Tensor:
    u = torch.where(valid, raw_u, torch.zeros_like(raw_u))
    if not coalition_norm:
        return u
    s_ind  = u.abs().sum(dim=-1, keepdim=True)
    target = torch.maximum(u_all.abs().unsqueeze(-1), 0.25 * s_ind)
    g_hi   = max(0.25, float(gamma_max))
    gamma  = (target / s_ind.clamp_min(1e-6)).clamp(0.25, g_hi)
    gamma  = torch.where((s_ind > 1e-6) & usable_all.unsqueeze(-1), gamma, torch.ones_like(gamma))
    return u * gamma


@torch._dynamo.disable
def _compute_exact_prefix_jacobians(
    raw_model: nn.Module,
    interm_det: list[Tensor],
    tracked_layers: list[int],
    b_cal: Tensor,
    pos_cal: Tensor,
) -> list[Tensor]:
    """
    Exact expected causal Jacobians `J_l = E[d x_{L,p} / d x_{l,p}]` (`[D_out, D_in]`, matching Anthropic 2026 J-Lens
    and `fit_toy_jacobian` in `jlens_aux_loss_demo.py`) across ALL `tracked_layers` in ONE single-position prefix-KV
    forward + batched VJP pass (`dgrad`-only via detached FP32 weights, zero parameter `.grad` mutation!).
    """
    dev      = interm_det[0].device
    T        = interm_det[0].shape[1]
    D        = raw_model.cfg.d_model
    H        = raw_model.cfg.n_heads
    Dh       = raw_model.cfg.head_dim
    n_layers = raw_model.cfg.n_layers
    min_l    = min(tracked_layers)
    B_cal    = b_cal.shape[0]
    inv_sq   = 1.0 / math.sqrt(float(Dh))

    with torch.no_grad(), torch.amp.autocast(device_type=dev.type, enabled=False):
        uniq_b, inv_b = torch.unique(b_cal, sorted=True, return_inverse=True)
        B_u = uniq_b.shape[0]
        cos_t = raw_model.cos[:, :, :T, :].detach().float()
        sin_t = raw_model.sin[:, :, :T, :].detach().float()
        kv_clean = []
        for m in range(min_l, n_layers):
            blk = raw_model.blocks[m]
            xb  = interm_det[m - 1][uniq_b].detach().float()
            an  = pure_rms_norm_fp32(xb, blk.attn_norm.eps) * blk.attn_norm.weight.detach().float()
            kv  = F.linear(an, blk.wqkv.weight[D:].detach().float()).view(B_u, T, 2, H, Dh)
            kn  = pure_rms_norm_fp32(kv[:, :, 0], blk.k_norm.eps) * blk.k_norm.weight.detach().float()
            kc  = apply_rope(kn.transpose(1, 2), cos_t, sin_t)[inv_b].contiguous()
            vc  = kv[:, :, 1].transpose(1, 2)[inv_b].contiguous()
            kv_clean.append((kc, vc))

        t_ar      = torch.arange(T, device=dev)[None, :]
        pre_valid = (t_ar < pos_cal[:, None])[:, None, None, :]
        cos_p     = raw_model.cos[0, 0][pos_cal][:, None, None, :].detach().float()
        sin_p     = raw_model.sin[0, 0][pos_cal][:, None, None, :].detach().float()
        eye_b     = torch.eye(D, dtype=torch.float32, device=dev).unsqueeze(1).expand(D, B_cal, D).contiguous()

    tracked_set = set(tracked_layers)
    tap_map: dict[int, Tensor] = {}
    with torch.enable_grad(), torch.amp.autocast(device_type=dev.type, enabled=False):
        x = interm_det[min_l - 1][b_cal, pos_cal].detach().float().requires_grad_(True)
        for idx_m, m in enumerate(range(min_l, n_layers)):
            if m in tracked_set:
                tap_map[m] = x
            blk = raw_model.blocks[m]
            k_pre, v_pre = kv_clean[idx_m]

            an  = pure_rms_norm_fp32(x, blk.attn_norm.eps) * blk.attn_norm.weight.detach().float()
            qkv = F.linear(an, blk.wqkv.weight.detach().float()).view(B_cal, 3, H, Dh)
            qw  = apply_rope(
                (pure_rms_norm_fp32(qkv[:, 0], blk.q_norm.eps) * blk.q_norm.weight.detach().float()).unsqueeze(2),
                cos_p, sin_p,
            )
            kw  = apply_rope(
                (pure_rms_norm_fp32(qkv[:, 1], blk.k_norm.eps) * blk.k_norm.weight.detach().float()).unsqueeze(2),
                cos_p, sin_p,
            )
            vw  = qkv[:, 2].unsqueeze(2)

            sc_pre = (torch.matmul(qw, k_pre.transpose(-1, -2)) * inv_sq).masked_fill(~pre_valid, -float("inf"))
            sc_loc = (qw * kw).sum(dim=-1, keepdim=True) * inv_sq
            probs  = torch.softmax(torch.cat([sc_pre, sc_loc], dim=-1), dim=-1)
            y_attn = (torch.matmul(probs[..., :T], v_pre) + probs[..., T:] * vw).reshape(B_cal, D)

            x  = x + F.linear(y_attn, blk.wo.weight.detach().float())
            mn = pure_rms_norm_fp32(x, blk.mlp_norm.eps) * blk.mlp_norm.weight.detach().float()
            g, u = F.linear(mn, blk.wgu.weight.detach().float()).chunk(2, dim=-1)
            x  = x + F.linear(F.silu(g) * u, blk.wdown.weight.detach().float())

        tap_list = [tap_map[l] for l in tracked_layers]
        grads = torch.autograd.grad(x, tap_list, grad_outputs=eye_b, is_grads_batched=True)
    return [g.detach().mean(dim=1).contiguous() for g in grads]


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
        select_mode: str = "greedy_cos",
        strength: float = 1.0,
        match_norm: bool = True,
        metric: str = "cov",
        j_mode: str = "ols_norm",
        norm_w_dir: bool = False,
        live_readout: bool = False,
        reuse_clean_nll: bool = False,
        loss_kind: str = "superpos_unlikelihood",
        reduction: str = "batch_sum",
        exp_beta: float = 0.0,
        exp_norm: bool = True,
        bystander_tau: float = 0.0,
        bystander_rel_tau: float = 0.0,
        bystander_w: float = 0.0,
        bystander_unscaled: bool = False,
        coalition_norm: bool = True,
        gamma_max: float = 4.0,
        hop_norm: bool = True,
        enable_early: bool = True,
        layer_mode: str = "fixed",
        pos_mode: str = "hard_easy",
        nll_cap: float = 0.0,
        live_target_head: bool = False,
        gold_w: float = 0.0,
        shared_w8_weight: float = 0.0,
        shared_w8_horizons: int = 8,
        shared_w8_k_vocab: int = 1536,
        unigram_path: Path | str | None = None,
        jac_samples: int = 16,
        detach_prev: bool = False,
        ridge: float = 0.10,
        ema: float = 0.50,
        pos_chunk_sz: int = 8192,
        seed: int = 42,
    ):
        assert 1 <= early_layer < anchor_layer < raw_model.cfg.n_layers
        assert layer_mode in ("fixed", "random_gap4", "random_anc_7_10", "random_3_11", "split_6_9", "split_3_11")
        assert select_mode in ("greedy_cos", "maxmin", "kmeans", "emeans")
        assert j_mode in ("ols_norm", "identity", "true_jacobian")
        assert reduction in ("batch_sum", "active_mean")
        self.cfg                = raw_model.cfg
        self.device             = device
        self.early_layer        = early_layer
        self.anchor_layer       = anchor_layer
        self.layer_mode         = layer_mode
        self.active_frac        = float(active_frac)
        self.probe_seqs         = probe_seqs
        self.pool_size          = pool_size
        self.k_dirs             = k_dirs
        self.window             = window
        self.max_cosine         = max_cosine
        self.select_mode        = select_mode
        self.strength           = strength
        self.match_norm         = match_norm
        self.metric             = metric
        self.j_mode             = j_mode
        self.norm_w_dir         = bool(norm_w_dir)
        self.live_readout       = bool(live_readout)
        self.reuse_clean_nll    = bool(reuse_clean_nll)
        self.loss_kind          = loss_kind
        self.reduction          = reduction
        self.exp_beta           = float(exp_beta)
        self.exp_norm           = bool(exp_norm)
        self.bystander_tau      = float(bystander_tau)
        self.bystander_rel_tau  = float(bystander_rel_tau)
        self.bystander_w        = float(bystander_w)
        self.bystander_unscaled = bool(bystander_unscaled)
        self.coalition_norm     = coalition_norm
        self.gamma_max          = float(gamma_max)
        self.hop_norm           = hop_norm
        self.enable_early       = enable_early
        self.pos_mode           = pos_mode
        self.nll_cap            = float(nll_cap)
        self.live_target_head   = bool(live_target_head) or bool(live_readout)
        self.gold_w             = float(gold_w)
        self.shared_w8_weight   = float(shared_w8_weight)
        self.shared_w8_horizons = int(shared_w8_horizons)
        v_lim = min(50257, self.cfg.vocab_pad)
        self.shared_w8_k_vocab  = min(int(shared_w8_k_vocab), v_lim)
        self.jac_samples        = max(2, int(jac_samples))
        self.detach_prev        = detach_prev
        self.ridge              = ridge
        self.ema                = ema
        self.pos_chunk_sz       = pos_chunk_sz
        self._initialized       = (j_mode == "identity" and self.shared_w8_weight <= 0.0)
        self._n_refreshes       = 0
        self.pair_rng           = torch.Generator(device="cpu")
        self.pair_rng.manual_seed(int(seed) + 1009)

        self.readout = FrozenSoftcapReadout(
            raw_model.out_norm.weight, raw_model.tok_emb.weight, softcap=self.cfg.softcap
        ).to(device)

        self.split_anc_layers: list[int] = []
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
            tracked_layers = sorted(
                ({l for pr in self.valid_layer_pairs for l in pr} | {early_layer, anchor_layer})
                if enable_early else {la for _, la in self.valid_layer_pairs}
            )
        elif layer_mode == "random_anc_7_10":
            if self.cfg.n_layers >= 12:
                anc_list = [7, 8, 9, 10]
            else:
                anc_list = list(range(max(2, early_layer + 1), self.cfg.n_layers))
            self.valid_layer_pairs = [(early_layer, la) for la in anc_list]
            tracked_layers = sorted((set(anc_list) | {early_layer, anchor_layer}) if enable_early else set(anc_list))
        elif layer_mode == "random_3_11":
            min_past_2 = min(3, max(1, self.cfg.n_layers - 1))
            anc_list = list(range(min_past_2, self.cfg.n_layers))
            self.valid_layer_pairs = [(min(early_layer, max(1, la - 1)), la) for la in anc_list]
            tracked_layers = sorted((set(anc_list) | {early_layer, anchor_layer}) if enable_early else set(anc_list))
        elif layer_mode == "split_6_9":
            if self.cfg.n_layers >= 12:
                self.split_anc_layers = [6, 7, 8, 9]
            else:
                self.split_anc_layers = list(range(max(1, self.cfg.n_layers // 2), self.cfg.n_layers))
            self.valid_layer_pairs = [(early_layer, la) for la in self.split_anc_layers]
            tracked_layers = sorted(
                (set(self.split_anc_layers) | {early_layer, anchor_layer})
                if enable_early else set(self.split_anc_layers)
            )
        elif layer_mode == "split_3_11":
            min_past_2 = min(3, max(1, self.cfg.n_layers - 1))
            self.split_anc_layers = list(range(min_past_2, self.cfg.n_layers))
            self.valid_layer_pairs = [(min(early_layer, max(1, la - 1)), la) for la in self.split_anc_layers]
            tracked_layers = sorted(
                (set(self.split_anc_layers) | {early_layer, anchor_layer})
                if enable_early else set(self.split_anc_layers)
            )
        else:
            self.valid_layer_pairs = [(early_layer, anchor_layer)]
            tracked_layers = sorted({early_layer, anchor_layer}) if enable_early else [anchor_layer]

        if self.shared_w8_weight > 0.0 and anchor_layer not in tracked_layers:
            tracked_layers = sorted(set(tracked_layers) | {anchor_layer})

        use_norm_in = (j_mode == "ols_norm")
        is_id       = (j_mode == "identity")
        self.tracked_layers = tracked_layers
        self.lenses = nn.ModuleDict({
            str(l): WorkspaceJLens(
                self.cfg.d_model, self.readout, device,
                norm_input=use_norm_in, is_identity=is_id, norm_w_dir=self.norm_w_dir,
            ).to(device)
            for l in tracked_layers
        })
        self.lens_early  = self.lenses[str(early_layer)] if str(early_layer) in self.lenses else None
        self.lens_anchor = (
            self.lenses[str(anchor_layer)] if str(anchor_layer) in self.lenses else self.lenses[str(tracked_layers[-1])]
        )

        w_vec = torch.tensor([RHO_HARMONIC ** m for m in range(window)], dtype=torch.float32, device=device)
        self.win_weights = w_vec / w_vec.sum()

        if self.shared_w8_weight > 0.0:
            H_w = self.shared_w8_horizons
            eye_d = torch.eye(self.cfg.d_model, dtype=torch.float32, device=device)
            self.J_anc_w8 = eye_d.unsqueeze(0).expand(H_w, self.cfg.d_model, self.cfg.d_model).clone()
            hw = torch.tensor([RHO_HARMONIC ** h for h in range(H_w)], dtype=torch.float32, device=device)
            self.w8_h_weights = hw / hw.sum()
            if unigram_path is not None and Path(unigram_path).is_file():
                cnts = np.load(unigram_path)[:v_lim].astype(np.float64)
            else:
                cnts = np.ones((v_lim,), dtype=np.float64)
            p_v = np.maximum(cnts, 1.0) ** 0.75
            p_v /= p_v.sum()
            self.w8_vocab_log_q = torch.from_numpy(np.log(np.maximum(p_v, 1e-12)).astype(np.float32)).to(device)
        else:
            self.J_anc_w8 = None
            self.w8_h_weights = None
            self.w8_vocab_log_q = None

    def _anchor_probe_dims(self) -> tuple[bool, bool, int]:
        inc_clean = not (self.reuse_clean_nll and self.window == 1 and self.live_target_head)
        inc_joint = bool(self.coalition_norm)
        p_anc     = self.k_dirs + (1 if inc_clean else 0) + (1 if inc_joint else 0)
        return inc_clean, inc_joint, p_anc

    def estimate_refresh_flops(self, micro_batch: int) -> float:
        """Exact FLOPs for one `refresh_snapshot` call (`true_jacobian` VJP, `ols_norm` Cholesky, `cov`, `shared_w8`)."""
        T   = float(self.cfg.seq_len)
        D   = float(self.cfg.d_model)
        F_d = float(self.cfg.d_ff)
        blk_lin_2n = 2.0 * (D * 3.0 * D + D * D + D * 2.0 * F_d + F_d * D)
        fl = 0.0
        if self.j_mode == "true_jacobian":
            n_j_blk = float(self.cfg.n_layers - min(self.tracked_layers))
            b_c = float(self.jac_samples)
            # Single-pos fwd + batched VJP (D dgrad passes) + prefix 2*D KV + batched VJP attention 4*(1+D)*b_c*T*D:
            fl += n_j_blk * ((1.0 + D) * b_c * blk_lin_2n + 2.0 * b_c * T * D * (2.0 * D) + 4.0 * (1.0 + D) * b_c * T * D)
        use_sh_w8 = (self.shared_w8_weight > 0.0) and (self.cfg.seq_len > self.shared_w8_horizons)
        if self.j_mode == "ols_norm" or use_sh_w8 or self.metric == "cov":
            k_r   = float(min(4096, micro_batch * self.cfg.seq_len))
            n_trk = float(len(self.tracked_layers))
            if self.j_mode == "ols_norm" or use_sh_w8:
                fl += n_trk * (2.0 * k_r * D * D + (D ** 3) / 3.0)
                if self.j_mode == "ols_norm":
                    fl += n_trk * (2.0 * k_r * D * D + 2.0 * (D ** 3))
                if use_sh_w8:
                    h_w = float(self.shared_w8_horizons)
                    fl += 2.0 * k_r * h_w * D * D + 2.0 * h_w * (D ** 3)
            if self.metric == "cov":
                fl += n_trk * (2.0 * k_r * D * D)
        return fl

    def estimate_call_flops(
        self,
        micro_batch: int,
        n_act: int | None = None,
        n_uniq_seqs: int | None = None,
        early_layer: int | None = None,
        anchor_layer: int | None = None,
        did_refresh: bool = False,
    ) -> float:
        """Exact FLOPs per `compute_loss` call including 4-GEMM readout, `select_diverse_from_pool`, & refresh VJP/OLS."""
        T   = self.cfg.seq_len
        D   = self.cfg.d_model
        F_d = self.cfg.d_ff
        V   = self.cfg.vocab_pad
        _, inc_joint, P_anc = self._anchor_probe_dims()
        P_ear = self.k_dirs + 1 + (1 if inc_joint else 0)
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
        if self.layer_mode in ("split_6_9", "split_3_11") and not self.enable_early and self.split_anc_layers:
            n_kv_blk  = float(self.cfg.n_layers - min(self.split_anc_layers))
            n_anc_blk = sum(float(self.cfg.n_layers - l) for l in self.split_anc_layers) / float(len(self.split_anc_layers))
            n_ear_blk = 0.0
        else:
            n_anc_blk = float(self.cfg.n_layers - anc_l)
            n_ear_blk = float(anc_l - ear_l) if self.enable_early else 0.0
            n_kv_blk  = n_anc_blk + n_ear_blk

        kv_proj_cols = (2 * D) if self.cfg.linear_mode == "bf16" else (3 * D)
        kv_flops = n_kv_blk * (2.0 * n_uniq_seqs * T * D * kv_proj_cols)

        blk_lin_2n = 2.0 * (D * 3 * D + D * D + D * 2 * F_d + F_d * D)
        anc_tok = n_act * P_anc * W_a
        anc_flops = n_anc_blk * (anc_tok * blk_lin_2n + 4.0 * anc_tok * (0.5 * T + W_a) * D)
        ear_tok = n_act * P_ear * 1
        ear_flops = n_ear_blk * (ear_tok * blk_lin_2n + 4.0 * ear_tok * (0.5 * T + 1) * D)

        j_mat_cost = 0.0 if self.j_mode == "identity" else (D * D)
        # 4 readout GEMMs per active layer: topk_pool (2NDV) + selected fwd (2NDV) + bwd recompute (2NDV) + bwd dgrad (2NDV) = 8NDV
        ro_anc = 2.0 * (anc_tok * D * V) + 6.0 * (n_act * j_mat_cost) + 8.0 * (n_act * D * V)
        ro_ear = (2.0 * (ear_tok * (j_mat_cost + D * V)) + 6.0 * (n_act * j_mat_cost) + 8.0 * (n_act * D * V)) if self.enable_early else 0.0

        # `select_diverse_from_pool` + joint oblique Gram (previously uncounted):
        M_p = float(min(self.pool_size, V))
        k_f = float(self.k_dirs)
        n_sel_layers = 2.0 if self.enable_early else 1.0
        sel_uj   = 0.0 if self.j_mode == "identity" else (2.0 * n_act * M_p * D * D)
        sel_sig  = (2.0 * n_act * M_p * D * D) if self.metric == "cov" else 0.0
        sel_gram = 2.0 * n_act * M_p * M_p * D
        sel_km   = (14.0 * n_act * M_p * M_p * k_f) if self.select_mode in ("kmeans", "emeans") else 0.0
        jnt_gram = (2.0 * n_act * k_f * k_f * D) if inc_joint else 0.0
        sel_fl   = n_sel_layers * (sel_uj + sel_sig + sel_gram + sel_km + jnt_gram)

        sh_w8 = (
            4.0 * float(n_act) * float(self.shared_w8_horizons) * D * (D + float(self.shared_w8_k_vocab) + 1.0)
            if self.shared_w8_weight > 0.0 else 0.0
        )
        ref_fl = self.estimate_refresh_flops(micro_batch) if did_refresh else 0.0
        return kv_flops + anc_flops + ear_flops + ro_anc + ro_ear + sel_fl + sh_w8 + ref_fl

    @torch.no_grad()
    def refresh_snapshot(self, raw_model: nn.Module, interm_det: list[Tensor], x_raw_final: Tensor, yb: Tensor | None = None):
        with torch.amp.autocast(device_type=x_raw_final.device.type, enabled=False):
            # Bias-corrected EMA warmup: exact running mean for early refreshes, transitioning smoothly to `self.ema`:
            alpha = min(self.ema, self._n_refreshes / float(self._n_refreshes + 1)) if self._initialized else 0.0
            if self.live_readout:
                self.readout.copy_from_(raw_model.out_norm.weight, raw_model.tok_emb.weight)
            else:
                self.readout.update_ema_(raw_model.out_norm.weight, raw_model.tok_emb.weight, ema=alpha)

            B, T, D = x_raw_final.shape
            M_tot   = B * T
            use_sh_w8 = (self.shared_w8_weight > 0.0) and (T > self.shared_w8_horizons)

            if self.j_mode == "identity" and not use_sh_w8 and self.metric != "cov":
                self._initialized = True
                self._n_refreshes += 1
                return

            if use_sh_w8:
                H_w  = self.shared_w8_horizons
                T_v  = T - (H_w - 1)
                K_r  = min(4096, B * T_v)
                b_r  = torch.randint(0, B, (K_r,), device=x_raw_final.device)
                t_r  = torch.randint(0, T_v, (K_r,), device=x_raw_final.device)
                ridx = b_r * T + t_r
            else:
                K_r  = min(4096, M_tot)
                ridx = torch.randint(0, M_tot, (K_r,), device=x_raw_final.device)

            inv_k = 1.0 / float(K_r)
            eye   = torch.eye(D, dtype=torch.float32, device=x_raw_final.device).unsqueeze(0)

            if self.j_mode == "true_jacobian":
                b_c = torch.randint(0, B, (self.jac_samples,), device=x_raw_final.device)
                p_c = torch.randint(min(1, T - 1), T, (self.jac_samples,), device=x_raw_final.device)
                j_list = _compute_exact_prefix_jacobians(raw_model, interm_det, self.tracked_layers, b_c, p_c)
                for l_idx, j_mat in zip(self.tracked_layers, j_list):
                    lens = self.lenses[str(l_idx)]
                    lens.J.mul_(alpha).add_(j_mat, alpha=(1.0 - alpha))

            if self.j_mode == "ols_norm" or use_sh_w8 or self.metric == "cov":
                y_fin = pure_rms_norm_fp32(x_raw_final.reshape(M_tot, D)[ridx])
                x_stk = torch.stack(
                    [pure_rms_norm_fp32(interm_det[l_idx - 1].reshape(M_tot, D)[ridx]) for l_idx in self.tracked_layers],
                    dim=0,
                )
                if self.j_mode == "ols_norm" or use_sh_w8:
                    g_xx = torch.matmul(x_stk.transpose(-1, -2), x_stk) * inv_k
                    tr_g = g_xx.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).clamp_min(1e-6).unsqueeze(-1)
                    chol = torch.linalg.cholesky(g_xx + (self.ridge * tr_g) * eye)
                    if self.j_mode == "ols_norm":
                        cxy   = torch.einsum("lkd,ke->lde", x_stk, y_fin) * inv_k
                        j_all = torch.cholesky_solve(cxy, chol).transpose(-1, -2).contiguous()
                        for idx_l, l_idx in enumerate(self.tracked_layers):
                            self.lenses[str(l_idx)].J.mul_(alpha).add_(j_all[idx_l], alpha=(1.0 - alpha))
                    if use_sh_w8:
                        anc_pos = self.tracked_layers.index(self.anchor_layer)
                        ridx_h  = ridx[:, None] + torch.arange(H_w, device=x_raw_final.device)[None, :]
                        if yb is not None:
                            vw_win = torch.cumprod((yb.reshape(-1)[ridx_h] != EOT_TOKEN).float(), dim=-1)
                        else:
                            vw_win = torch.ones((K_r, H_w), dtype=torch.float32, device=x_raw_final.device)
                        y_fh   = pure_rms_norm_fp32(x_raw_final.reshape(M_tot, D)[ridx_h]) * vw_win.unsqueeze(-1)
                        cnt_h  = vw_win.sum(dim=0).clamp_min(1.0).view(1, H_w, 1)
                        cxy_w8 = (torch.einsum("kd,khf->dhf", x_stk[anc_pos], y_fh) / cnt_h).reshape(D, H_w * D)
                        w_star = torch.cholesky_solve(cxy_w8, chol[anc_pos]).view(D, H_w, D).permute(1, 0, 2).contiguous()
                        self.J_anc_w8.mul_(alpha).add_(w_star, alpha=(1.0 - alpha))

                if self.metric == "cov":
                    x_c = x_stk - x_stk.mean(dim=1, keepdim=True)
                    cov = torch.matmul(x_c.transpose(-1, -2), x_c) * inv_k
                    tr_c = cov.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).clamp_min(1e-6).unsqueeze(-1)
                    sig_all = cov / tr_c + 0.02 * eye
                    for idx_l, l_idx in enumerate(self.tracked_layers):
                        self.lenses[str(l_idx)].Sigma.mul_(alpha).add_(sig_all[idx_l], alpha=(1.0 - alpha))

            self._initialized = True
            self._n_refreshes += 1

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
                if self.nll_cap > 0.0:
                    capped_sc = torch.where(nll_flat <= self.nll_cap, nll_flat, -nll_flat)
                    nll_h = torch.where(valid_flat, capped_sc, torch.full_like(nll_flat, -1e9))
                else:
                    nll_h = torch.where(valid_flat, nll_flat, torch.full_like(nll_flat, -1e9))
                _, hard_pool = torch.topk(nll_h, k_pool_hard, largest=True)

                is_hard = torch.zeros_like(valid_flat)
                is_hard[hard_pool] = True
                easy_elig = valid_flat & (~is_hard) & (nll_flat > 0.05)
                min_easy_pool = max(2, int(math.ceil(max(0.05, (1.0 / 3.0) * self.active_frac) * M_tot)))
                if int(easy_elig.sum().item()) < min_easy_pool:
                    easy_elig = valid_flat & (~is_hard)
                n_easy_elig = int(easy_elig.sum().item())
                if n_easy_elig > 0:
                    k_pool_easy = min(n_easy_elig, min_easy_pool)
                    nll_e = torch.where(easy_elig, nll_flat, torch.full_like(nll_flat, 1e9))
                    _, easy_pool = torch.topk(nll_e, k_pool_easy, largest=False)
                else:
                    easy_pool = torch.empty((0,), dtype=torch.long, device=yb.device)

                n_take_hard = min(hard_pool.numel(), int(round((2.0 / 3.0) * n_act_tot)))
                n_take_easy = min(easy_pool.numel(), n_act_tot - n_take_hard)
                if n_take_hard + n_take_easy < n_act_tot:
                    n_take_hard = min(hard_pool.numel(), n_act_tot - n_take_easy)

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

    def _eval_anchor_group(
        self,
        raw_model: nn.Module,
        interm: list[Tensor],
        cur_anc: int,
        kv_anc: list[tuple[Tensor, Tensor]],
        b_each: Tensor,
        pos_each: Tensor,
        kv_slot: Tensor,
        yb: Tensor,
        tok_nll_2d: Tensor,
        sup_fn,
        adv_fn,
        live_nw: Tensor | None,
        live_u_bf: Tensor | None,
        live_u_fp: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Evaluates 1 anchor layer `cur_anc` on sorted `(b_each, pos_each)` returning `(vec_anc, ids_a, u_anc, usable_a, val_a)`."""
        W = self.window
        k = self.k_dirs
        inc_clean, inc_joint, P_anc = self._anchor_probe_dims()
        N_g  = b_each.shape[0]
        c_sz = self.pos_chunk_sz
        lens_anc = self.lenses[str(cur_anc)]

        h_anc_seq = interm[cur_anc - 1]
        if self.detach_prev and cur_anc >= 2:
            prev_a = interm[cur_anc - 2].detach()
            h_anc_seq = prev_a + (h_anc_seq - prev_a)
        x_anc_live = h_anc_seq[b_each, pos_each]

        pool_a, probs_a = lens_anc.topk_pool_ids_and_probs(x_anc_live, pool_size=self.pool_size)
        ids_a, w_a, d_a, val_a = select_diverse_from_pool(
            pool_a, lens_anc, k=k, max_cosine=self.max_cosine, metric=self.metric,
            select_mode=self.select_mode, pool_probs=probs_a,
        )

        with torch.no_grad():
            anc_blocks = raw_model.blocks[cur_anc :]
            w_offs = torch.arange(W, device=yb.device)
            u_anc_parts = []
            usable_a_parts = []
            h_anc_det = interm[cur_anc - 1].detach()

            for s in range(0, N_g, c_sz):
                e = min(s + c_sz, N_g)
                nc = e - s
                probes_c, usa_c, usa_all_c = sup_fn(
                    x_anc_live[s:e].detach(), w_a[s:e], d_a[s:e], val_a[s:e],
                    strength=self.strength, match_norm=self.match_norm,
                    include_clean=inc_clean, include_joint=inc_joint,
                )
                win_idx_c = pos_each[s:e, None] + w_offs[None, :]
                h_clean_c = h_anc_det[b_each[s:e, None], win_idx_c]
                h_win_c   = h_clean_c.unsqueeze(1).expand(nc, P_anc, W, self.cfg.d_model).clone()
                h_win_c[:, :, 0, :] = probes_c

                fin_c = adv_fn(
                    anc_blocks, kv_anc, h_win_c, pos_each[s:e], raw_model.cos, raw_model.sin, kv_slot=kv_slot[s:e]
                )
                y_win_c  = yb[b_each[s:e, None], win_idx_c]
                flat_fin = fin_c.reshape(-1, self.cfg.d_model)
                flat_y   = y_win_c[:, None, :].expand(nc, P_anc, W).reshape(-1)
                nll_pw   = self.readout.fast_nograd_target_nll(
                    flat_fin, flat_y, norm_w_override=live_nw, U_bf16_override=live_u_bf, U_fp32_override=live_u_fp
                ).view(nc, P_anc, W)

                valid_y = (y_win_c != EOT_TOKEN).float()
                wt_y = valid_y * self.win_weights.unsqueeze(0)
                wt_y = wt_y / wt_y.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                scores_c = (nll_pw * wt_y.unsqueeze(1)).sum(dim=-1)

                if inc_clean:
                    clean_sc = scores_c[:, 0:1]
                    k_off    = 1
                else:
                    clean_sc = (tok_nll_2d[b_each[s:e], pos_each[s:e]].float() * valid_y[:, 0]).unsqueeze(1)
                    k_off    = 0

                raw_u_c = scores_c[:, k_off : k_off + k] - clean_sc
                if inc_joint:
                    u_all_c = scores_c[:, k_off + k] - clean_sc.squeeze(1)
                else:
                    u_all_c = torch.zeros((nc,), dtype=torch.float32, device=yb.device)
                u_anc_parts.append(calibrate_usefulness(raw_u_c, u_all_c, usa_c, usa_all_c, self.coalition_norm, self.gamma_max))
                usable_a_parts.append(usa_c)

            u_anc    = torch.cat(u_anc_parts, dim=0)
            usable_a = torch.cat(usable_a_parts, dim=0)

        y_gold = yb[b_each, pos_each]
        if self.gold_w > 0.0:
            ids_plus_gold = torch.cat([ids_a, y_gold.unsqueeze(1)], dim=1)
            lq_all_anc    = lens_anc.selected_log_probs(x_anc_live, ids_plus_gold)
            lq_sel_anc    = lq_all_anc[:, :k]
            lq_gold_anc   = lq_all_anc[:, k]
            gold_vec      = -float(self.gold_w) * lq_gold_anc.float() * (y_gold != EOT_TOKEN).float()
        else:
            lq_sel_anc    = lens_anc.selected_log_probs(x_anc_live, ids_a)
            gold_vec      = None

        u_cl_lim = 5.0 if self.loss_kind == "exp_unlikelihood" else 0.6
        u_abs_cl = u_anc.clamp(-u_cl_lim, u_cl_lim).abs()
        if self.bystander_rel_tau > 0.0:
            mean_ul = float((u_abs_cl * usable_a.float()).sum().item()) / max(1.0, float(usable_a.float().sum().item()))
            tau_eff = max(self.bystander_tau, self.bystander_rel_tau * mean_ul)
        else:
            tau_eff = self.bystander_tau

        vec_anc = compute_feature_loss_pregathered(
            lq_sel_anc, u_anc, usable_a, loss_kind=self.loss_kind, exp_beta=self.exp_beta,
            exp_norm=self.exp_norm, bystander_tau=tau_eff, bystander_w=self.bystander_w,
            bystander_unscaled=self.bystander_unscaled,
        )
        if gold_vec is not None:
            vec_anc = vec_anc + gold_vec
        bys_mask_a = (u_abs_cl < tau_eff) & usable_a if tau_eff > 0.0 else torch.zeros_like(usable_a)
        return vec_anc, ids_a, u_anc, usable_a, val_a, bys_mask_a

    @torch._dynamo.disable
    def compute_loss(
        self,
        raw_model: nn.Module,
        interm: list[Tensor],
        yb: Tensor,
        tok_nll_det: Tensor,
    ) -> tuple[Tensor, Tensor, dict]:
        k = self.k_dirs
        B_tot, T_tot = yb.shape
        inv_m_tot = 1.0 / float(B_tot * T_tot)
        tok_nll_2d = tok_nll_det.view(B_tot, T_tot)

        if self.live_readout:
            self.readout.copy_from_(raw_model.out_norm.weight, raw_model.tok_emb.weight)

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

        if self.live_readout:
            live_nw = live_u_bf = live_u_fp = None
        elif self.live_target_head:
            live_nw   = raw_model.out_norm.weight.detach().float()
            live_u_bf = raw_model.tok_emb.weight.detach().to(torch.bfloat16)
            live_u_fp = None if use_fast_gpu else raw_model.tok_emb.weight.detach().float()
        else:
            live_nw = live_u_bf = live_u_fp = None

        # --- Multi-Layer Shared-KV Split-Batch Mode (`split_6_9` / `split_3_11`, `enable_early == False`) ---
        if self.layer_mode in ("split_6_9", "split_3_11") and not self.enable_early and self.split_anc_layers:
            min_anc = min(self.split_anc_layers)
            with torch.no_grad():
                kv_base = kv_fn(
                    raw_model.blocks[min_anc :],
                    [interm[m - 1].detach() for m in range(min_anc, self.cfg.n_layers)],
                    uniq_b, raw_model.cos, raw_model.sin,
                )
                perm = torch.randperm(N_act, device=yb.device)
                chunks = torch.chunk(perm, len(self.split_anc_layers))

            vec_parts, u_parts, usa_parts, val_parts, bys_parts = [], [], [], [], []
            for idx_c, l_anc in zip(chunks, self.split_anc_layers):
                if idx_c.numel() == 0:
                    continue
                # Sorting `idx_c` preserves strict ascending `(b_each, pos_each)` and monotonic `kv_slot` for `build_seq_groups`!
                sub = idx_c.sort()[0]
                v_l, _, u_l, us_l, vl_l, bm_l = self._eval_anchor_group(
                    raw_model, interm, l_anc, kv_base[l_anc - min_anc :],
                    b_each[sub], pos_each[sub], kv_slot[sub], yb, tok_nll_2d,
                    sup_fn, adv_fn, live_nw, live_u_bf, live_u_fp,
                )
                vec_parts.append(v_l)
                u_parts.append(u_l)
                usa_parts.append(us_l)
                val_parts.append(vl_l)
                bys_parts.append(bm_l)
            del kv_base

            vec_anc        = torch.cat(vec_parts, dim=0)
            u_anc          = torch.cat(u_parts, dim=0)
            usable_a       = torch.cat(usa_parts, dim=0)
            val_a          = torch.cat(val_parts, dim=0)
            bys_mask_a     = torch.cat(bys_parts, dim=0)
            loss_anc_mean  = torch.stack([v.mean() for v in vec_parts]).mean()
            loss_anc_batch = vec_anc.sum() * inv_m_tot
            cur_ear, cur_anc = self.early_layer, self.anchor_layer
            ids_a = None
        else:
            if self.layer_mode in ("random_gap4", "random_anc_7_10", "random_3_11"):
                p_idx = int(torch.randint(0, len(self.valid_layer_pairs), (1,), generator=self.pair_rng).item())
                cur_ear, cur_anc = self.valid_layer_pairs[p_idx]
            else:
                cur_ear, cur_anc = self.early_layer, self.anchor_layer

            with torch.no_grad():
                anc_blocks = raw_model.blocks[cur_anc :]
                anc_inputs = [interm[m - 1].detach() for m in range(cur_anc, self.cfg.n_layers)]
                kv_anc = kv_fn(anc_blocks, anc_inputs, uniq_b, raw_model.cos, raw_model.sin)

            vec_anc, ids_a, u_anc, usable_a, val_a, bys_mask_a = self._eval_anchor_group(
                raw_model, interm, cur_anc, kv_anc,
                b_each, pos_each, kv_slot, yb, tok_nll_2d,
                sup_fn, adv_fn, live_nw, live_u_bf, live_u_fp,
            )
            del kv_anc

            if self.shared_w8_weight > 0.0 and self.J_anc_w8 is not None:
                h_anc_seq  = interm[cur_anc - 1]
                x_anc_live = h_anc_seq[b_each, pos_each]
                H_w    = self.shared_w8_horizons
                h_offs = torch.arange(H_w, device=yb.device)[None, :]
                t_kh   = (pos_each[:, None] + h_offs).clamp_max(T_tot - 1)
                in_b   = (pos_each[:, None] + h_offs) < T_tot
                y_kh   = yb[b_each[:, None], t_kh]
                val_kh = torch.cumprod(((y_kh != EOT_TOKEN) & in_b).float(), dim=-1)
                eff_wh = val_kh * self.w8_h_weights.unsqueeze(0)
                eff_wh = eff_wh / eff_wh.sum(dim=-1, keepdim=True).clamp_min(1e-6)

                with torch.no_grad():
                    u_g = torch.empty_like(self.w8_vocab_log_q).uniform_(1e-7, 1.0 - 1e-7)
                    _, c_ids = torch.topk(self.w8_vocab_log_q - torch.log(-torch.log(u_g)), self.shared_w8_k_vocab)
                    log_kc = math.log(float(self.shared_w8_k_vocab)) + self.w8_vocab_log_q[c_ids]

                xn_w8 = pure_rms_norm_fp32(x_anc_live).to(torch.bfloat16)
                j_w8  = self.J_anc_w8.detach().to(torch.bfloat16)
                nw_bf = self.readout.norm_w.detach().to(torch.bfloat16)
                h_nh  = (torch.einsum("nd,hdo->nho", xn_w8, j_w8) * nw_bf[None, None, :]).reshape(N_act * H_w, self.cfg.d_model)
                u_bf  = self.readout.U_bf16.detach()
                y_fl  = y_kh.reshape(N_act * H_w)
                nll_fn = _compiled_sampled_softcap_nll_vec if use_fast_gpu else _sampled_softcap_nll_vec
                nll_nh = nll_fn(h_nh, u_bf[y_fl], u_bf[c_ids], y_fl, c_ids, log_kc, float(self.cfg.softcap)).view(N_act, H_w)
                w8_vec = (nll_nh * eff_wh).sum(dim=-1) * val_kh[:, 0]
                vec_anc = vec_anc + float(self.shared_w8_weight) * w8_vec

            loss_anc_mean  = vec_anc.mean()
            loss_anc_batch = vec_anc.sum() * inv_m_tot

        # --- 2. Early-Layer Telescoped Probes (`cur_ear .. cur_anc`, `W_early = 1` at `pos_each`!) ---
        loss_ear_mean  = torch.zeros((), dtype=torch.float32, device=yb.device)
        loss_ear_batch = torch.zeros((), dtype=torch.float32, device=yb.device)
        u_ear_abs = 0.0
        if self.enable_early and ids_a is not None:
            lens_ear = self.lenses[str(cur_ear)]
            lens_anc = self.lenses[str(cur_anc)]
            inc_joint = bool(self.coalition_norm)
            P_ear = k + 1 + (1 if inc_joint else 0)

            h_ear_seq = interm[cur_ear - 1]
            if self.detach_prev and cur_ear >= 2:
                prev_e = interm[cur_ear - 2].detach()
                h_ear_seq = prev_e + (h_ear_seq - prev_e)
            x_ear_live = h_ear_seq[b_each, pos_each]

            pool_e, probs_e = lens_ear.topk_pool_ids_and_probs(x_ear_live, pool_size=self.pool_size)
            ids_e, w_e, d_e, val_e = select_diverse_from_pool(
                pool_e, lens_ear, k=k, max_cosine=self.max_cosine, metric=self.metric,
                select_mode=self.select_mode, pool_probs=probs_e,
            )

            with torch.no_grad():
                ear_blocks = raw_model.blocks[cur_ear : cur_anc]
                ear_inputs = [interm[m - 1].detach() for m in range(cur_ear, cur_anc)]
                kv_ear = kv_fn(ear_blocks, ear_inputs, uniq_b, raw_model.cos, raw_model.sin)

                # Compute fixed anchor target weights BEFORE `hop_norm` so bystander `tau` is evaluated in true nat units
                # and `sc_ec` never double-exponentiates `exp_beta`!
                if self.loss_kind in ("exp_signed_clamped", "exp_unlikelihood"):
                    is_eul = (self.loss_kind == "exp_unlikelihood")
                    u_cl_a = 5.0 if is_eul else 0.6
                    if self.bystander_rel_tau > 0.0:
                        mean_ua = float((u_anc.clamp(-u_cl_a, u_cl_a).abs() * usable_a.float()).sum().item()) / max(1.0, float(usable_a.float().sum().item()))
                        tau_anc_eff = max(self.bystander_tau, self.bystander_rel_tau * mean_ua)
                    else:
                        tau_anc_eff = self.bystander_tau
                    u_eff_anc = compute_effective_feature_weights(
                        u_anc, usable_a, exp_beta=self.exp_beta, exp_norm=self.exp_norm,
                        bystander_tau=tau_anc_eff, bystander_w=self.bystander_w,
                        bystander_unscaled=self.bystander_unscaled,
                        u_clamp=u_cl_a, shift_neg_by_bys=(not is_eul),
                    )
                else:
                    u_eff_anc = u_anc
                if self.hop_norm:
                    norm_a = u_eff_anc.abs().sum(dim=-1, keepdim=True).clamp_min(0.05)
                    u_anc_target = u_eff_anc / norm_a
                else:
                    u_anc_target = u_eff_anc

                u_ear_parts = []
                usable_e_parts = []
                for s in range(0, N_act, c_sz):
                    e = min(s + c_sz, N_act)
                    nc = e - s
                    probes_ec, usa_ec, usa_all_ec = sup_fn(
                        x_ear_live[s:e].detach(), w_e[s:e], d_e[s:e], val_e[s:e],
                        strength=self.strength, match_norm=self.match_norm,
                        include_clean=True, include_joint=inc_joint,
                    )
                    at_anc_c = adv_fn(
                        ear_blocks, kv_ear, probes_ec.unsqueeze(2), pos_each[s:e],
                        raw_model.cos, raw_model.sin, kv_slot=kv_slot[s:e],
                    ).squeeze(2)

                    with torch.amp.autocast(device_type=yb.device.type, enabled=False):
                        at_anc_h = lens_anc.project_hidden(at_anc_c.reshape(nc * P_ear, self.cfg.d_model))
                    ids_ac = ids_a[s:e].unsqueeze(1).expand(nc, P_ear, k).reshape(nc * P_ear, k)
                    ut_c   = u_anc_target[s:e].unsqueeze(1).expand(nc, P_ear, k).reshape(nc * P_ear, k)
                    usa_ac = usable_a[s:e].unsqueeze(1).expand(nc, P_ear, k).reshape(nc * P_ear, k)
                    lq_ec  = self.readout.fast_nograd_selected_logprobs(at_anc_h, ids_ac)
                    sc_ec  = compute_feature_loss_pregathered(
                        lq_ec, ut_c, usa_ac, loss_kind=self.loss_kind,
                        exp_beta=0.0, exp_norm=False, bystander_tau=0.0, bystander_w=0.0,
                    ).view(nc, P_ear)

                    raw_u_ec = sc_ec[:, 1 : k + 1] - sc_ec[:, 0:1]
                    u_all_ec = (sc_ec[:, k + 1] - sc_ec[:, 0]) if inc_joint else torch.zeros((nc,), dtype=torch.float32, device=yb.device)
                    u_ear_parts.append(calibrate_usefulness(raw_u_ec, u_all_ec, usa_ec, usa_all_ec, self.coalition_norm, self.gamma_max))
                    usable_e_parts.append(usa_ec)

                u_ear    = torch.cat(u_ear_parts, dim=0)
                usable_e = torch.cat(usable_e_parts, dim=0)
                u_ear_abs = float(u_ear.abs().mean().item())
                del kv_ear

            u_cl_e = 5.0 if self.loss_kind == "exp_unlikelihood" else 0.6
            if self.bystander_rel_tau > 0.0:
                mean_ue = float((u_ear.clamp(-u_cl_e, u_cl_e).abs() * usable_e.float()).sum().item()) / max(1.0, float(usable_e.float().sum().item()))
                tau_ear_eff = max(self.bystander_tau, self.bystander_rel_tau * mean_ue)
            else:
                tau_ear_eff = self.bystander_tau

            lq_sel_ear     = lens_ear.selected_log_probs(x_ear_live, ids_e)
            vec_ear        = compute_feature_loss_pregathered(
                lq_sel_ear, u_ear, usable_e, loss_kind=self.loss_kind, exp_beta=self.exp_beta,
                exp_norm=self.exp_norm, bystander_tau=tau_ear_eff, bystander_w=self.bystander_w,
                bystander_unscaled=self.bystander_unscaled,
            )
            loss_ear_mean  = vec_ear.mean()
            loss_ear_batch = vec_ear.sum() * inv_m_tot

        if self.reduction == "active_mean":
            tot_aux = (loss_anc_mean + loss_ear_mean) if self.enable_early else loss_anc_mean
            tot_obs = tot_aux.detach()
        else:
            tot_aux = 0.5 * (loss_anc_batch + loss_ear_batch) if self.enable_early else loss_anc_batch
            tot_obs = (0.5 * (loss_anc_mean + loss_ear_mean) if self.enable_early else loss_anc_mean).detach()
        n_usa_a  = max(1.0, float(usable_a.float().sum().item()))
        bys_frac = float(bys_mask_a.float().sum().item()) / n_usa_a
        stats = {
            "l_ear": int(cur_ear),
            "l_anc": int(cur_anc),
            "n_act": int(N_act),
            "n_seqs": int(B_u),
            "k_filled": round(float(val_a.float().sum(dim=-1).mean().item()), 2),
            "bys_frac": round(bys_frac, 3),
            "u_anc_abs": round(float(u_anc.abs().mean().item()), 5),
            "u_anc_pos_frac": round(float((u_anc > 0).float().mean().item()), 3),
            "u_ear_abs": round(u_ear_abs, 5),
            "loss_anc": round(float(loss_anc_mean.detach().item()), 5),
            "loss_ear": round(float(loss_ear_mean.detach().item()), 5),
            "j_batch": round(float(tot_aux.detach().item()), 5),
        }
        return tot_aux, tot_obs, stats
