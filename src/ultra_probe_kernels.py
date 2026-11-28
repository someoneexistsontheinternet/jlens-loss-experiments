"""
Ultra-High-MFU Fused Probe-Window & Softcap-Bwd Kernels for `gfx942` (MI325X):
  1. `_probe_qkv_norm_rope_nhpwd_kernel`: Fuses `q_norm` + `k_norm` + RoPE + layout transpose
     `[N_pos * PW, 3, H, 64] -> 3 x contiguous [N_pos, H, PW, 64]` in ONE kernel (eliminates ~200 eager launches/chunk).
  2. `_grouped_flash_prefix_and_local_kernel`: Fuses grouped same-sequence `128x64` MFMA prefix FlashAttention
     AND the local `W x W` causal window (`tl.static_range(0, W)`) in registers, writing `y_attn [N_pos * P, W, D]`
     directly in BF16 (eliminates `3.2 GB` `o_pre_buf`, `lse_pre_buf`, `72` merge kernels/step, and `2.5 GB` transpose copy!).
  3. `_softcap_selected_bwd_2half_kernel`: 1-pass 2-half BF16-register selected-logprobs backward (`HALF_V=32768, num_warps=8`).
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import triton
import triton.language as tl

from fast_kernels import FusedRMSNormFn, FusedSwiGLUFn, build_seq_groups


@triton.jit
def _probe_qkv_norm_rope_nhpwd_kernel(
    qkv_ptr, pos_ptr, wq_ptr, wk_ptr, cos_ptr, sin_ptr,
    q_out_ptr, k_out_ptr, v_out_ptr,
    N_POS, PW: tl.constexpr, W: tl.constexpr, H: tl.constexpr, DH: tl.constexpr, HALF: tl.constexpr, EPS: tl.constexpr,
):
    """
    1 program per `(n_idx, pw_idx)` row (`N_pos * PW` programs):
    Reads contiguous `qkv [N_pos * PW, 3, H, 64]`, applies FP32 `q_norm`/`k_norm` + RoPE at `pos[n_idx] + (pw_idx % W)`,
    and writes contiguous `[N_pos, H, PW, 64]` for `q_pw, k_pw, v_pw`.
    """
    row = tl.program_id(0).to(tl.int64)
    n_idx  = row // PW
    pw_idx = row % PW
    wq_idx = pw_idx % W
    seq_t  = tl.load(pos_ptr + n_idx).to(tl.int64) + wq_idx

    offs_h = tl.arange(0, H).to(tl.int64)
    offs_d = tl.arange(0, HALF).to(tl.int64)
    offs_f = tl.arange(0, DH).to(tl.int64)

    in_base = qkv_ptr + row * (3 * H * DH)
    q1 = tl.load(in_base + 0 * (H * DH) + offs_h[:, None] * DH + offs_d[None, :]).to(tl.float32)
    q2 = tl.load(in_base + 0 * (H * DH) + offs_h[:, None] * DH + HALF + offs_d[None, :]).to(tl.float32)
    k1 = tl.load(in_base + 1 * (H * DH) + offs_h[:, None] * DH + offs_d[None, :]).to(tl.float32)
    k2 = tl.load(in_base + 1 * (H * DH) + offs_h[:, None] * DH + HALF + offs_d[None, :]).to(tl.float32)
    v  = tl.load(in_base + 2 * (H * DH) + offs_h[:, None] * DH + offs_f[None, :])

    wq1 = tl.load(wq_ptr + offs_d).to(tl.float32)[None, :]
    wq2 = tl.load(wq_ptr + HALF + offs_d).to(tl.float32)[None, :]
    wk1 = tl.load(wk_ptr + offs_d).to(tl.float32)[None, :]
    wk2 = tl.load(wk_ptr + HALF + offs_d).to(tl.float32)[None, :]

    cos = tl.load(cos_ptr + seq_t * HALF + offs_d).to(tl.float32)[None, :]
    sin = tl.load(sin_ptr + seq_t * HALF + offs_d).to(tl.float32)[None, :]

    inv_dh: tl.constexpr = 1.0 / DH
    rq = tl.math.rsqrt((tl.sum(q1 * q1, axis=1) + tl.sum(q2 * q2, axis=1)) * inv_dh + EPS)
    rk = tl.math.rsqrt((tl.sum(k1 * k1, axis=1) + tl.sum(k2 * k2, axis=1)) * inv_dh + EPS)

    qn1 = (q1 * rq[:, None] * wq1).to(tl.bfloat16).to(tl.float32)
    qn2 = (q2 * rq[:, None] * wq2).to(tl.bfloat16).to(tl.float32)
    kn1 = (k1 * rk[:, None] * wk1).to(tl.bfloat16).to(tl.float32)
    kn2 = (k2 * rk[:, None] * wk2).to(tl.bfloat16).to(tl.float32)

    oq1 = (qn1 * cos - qn2 * sin).to(tl.bfloat16)
    oq2 = (qn1 * sin + qn2 * cos).to(tl.bfloat16)
    ok1 = (kn1 * cos - kn2 * sin).to(tl.bfloat16)
    ok2 = (kn1 * sin + kn2 * cos).to(tl.bfloat16)

    # Output layout: contiguous [N_pos, H, PW, DH] -> offset = ((n_idx * H + h) * PW + pw_idx) * DH + d
    out_off = ((n_idx * H + offs_h) * PW + pw_idx)[:, None] * DH
    tl.store(q_out_ptr + out_off + offs_d[None, :], oq1)
    tl.store(q_out_ptr + out_off + HALF + offs_d[None, :], oq2)
    tl.store(k_out_ptr + out_off + offs_d[None, :], ok1)
    tl.store(k_out_ptr + out_off + HALF + offs_d[None, :], ok2)
    tl.store(v_out_ptr + out_off + offs_f[None, :], v)


@triton.jit
def _grouped_flash_prefix_and_local_kernel(
    q_ptr, k_clean_ptr, v_clean_ptr, k_loc_ptr, v_loc_ptr,
    grp_starts_ptr, grp_ends_ptr, kv_slot_ptr, pos_ptr,
    y_out_ptr,
    stride_qn, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kt,
    SM_SCALE,
    PW: tl.constexpr, W: tl.constexpr, H: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_T: tl.constexpr, DH: tl.constexpr,
):
    """
    Fuses:
      1) Grouped same-sequence `128x64` MFMA prefix FlashAttention (`0 .. pos[n_idx] - 1` on shared `K_clean, V_clean`)
      2) Local causal `W x W` window (`wk = 0 .. wq_idx` on probe-specific `k_loc, v_loc`)
      3) Direct BF16 output store into `y_out [N_pos * P, W, H * 64]` (ready for `blk.wo(y_out)` with ZERO intermediate buffers!).
    """
    pid_g = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)

    n_start = tl.load(grp_starts_ptr + pid_g).to(tl.int64)
    n_end   = tl.load(grp_ends_ptr + pid_g).to(tl.int64)
    cnt     = n_end - n_start
    valid_rows = cnt * PW

    offs_m = tl.arange(0, BLOCK_M).to(tl.int64)
    offs_d = tl.arange(0, DH).to(tl.int64)
    mask_m = offs_m < valid_rows

    n_idx  = n_start + (offs_m // PW)
    pw_idx = offs_m % PW
    p_idx  = pw_idx // W
    wq_idx = pw_idx % W

    q_ptrs = q_ptr + n_idx[:, None] * stride_qn + pid_h * stride_qh + pw_idx[:, None] * stride_qm + offs_d[None, :]
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.bfloat16)

    LOG2E: tl.constexpr = 1.4426950408889634
    qk_scale = SM_SCALE * LOG2E

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, DH), dtype=tl.float32)

    max_p_len = tl.load(pos_ptr + n_end - 1)
    if max_p_len > 0:
        b_slot  = tl.load(kv_slot_ptr + n_start).to(tl.int64)
        row_pos = tl.load(pos_ptr + n_idx, mask=mask_m, other=0)
        k_base  = k_clean_ptr + b_slot * stride_kb + pid_h * stride_kh
        v_base  = v_clean_ptr + b_slot * stride_kb + pid_h * stride_kh

        for t0 in range(0, max_p_len, BLOCK_T):
            offs_t = t0 + tl.arange(0, BLOCK_T)
            mask_t = offs_t < max_p_len
            k_tile = tl.load(k_base + offs_t[:, None] * stride_kt + offs_d[None, :], mask=mask_t[:, None], other=0.0).to(tl.bfloat16)
            qk = tl.dot(q, tl.trans(k_tile)) * qk_scale

            valid_kv   = offs_t[None, :] < row_pos[:, None]
            row_has_kv = t0 < row_pos
            qk = tl.where(valid_kv, qk, -float("inf"))

            blk_max = tl.max(qk, axis=1)
            m_new   = tl.where(row_has_kv, tl.maximum(m_i, blk_max), m_i)
            alpha   = tl.where(row_has_kv, tl.math.exp2(m_i - m_new), 1.0)
            m_safe  = tl.where(row_has_kv, m_new, 0.0)
            p_w     = tl.where(valid_kv, tl.math.exp2(qk - m_safe[:, None]), 0.0)

            l_i = l_i * alpha + tl.sum(p_w, axis=1)
            acc = acc * alpha[:, None]

            v_tile = tl.load(v_base + offs_t[:, None] * stride_kt + offs_d[None, :], mask=mask_t[:, None], other=0.0).to(tl.bfloat16)
            acc += tl.dot(p_w.to(tl.bfloat16), v_tile)
            m_i = m_new

    # Fused Local WxW Causal Window in Registers (`wk = 0 .. W - 1`, causal `wk <= wq_idx` within same probe `p_idx`!)
    qf = q.to(tl.float32)
    for wk in tl.static_range(0, W):
        loc_pw = p_idx * W + wk
        loc_off = n_idx[:, None] * stride_qn + pid_h * stride_qh + loc_pw[:, None] * stride_qm + offs_d[None, :]
        k_wk = tl.load(k_loc_ptr + loc_off, mask=mask_m[:, None], other=0.0).to(tl.float32)
        v_wk = tl.load(v_loc_ptr + loc_off, mask=mask_m[:, None], other=0.0).to(tl.float32)

        s_wk = tl.sum(qf * k_wk, axis=1) * qk_scale
        is_ok = mask_m & (wk <= wq_idx)
        s_msk = tl.where(is_ok, s_wk, -float("inf"))

        m_new     = tl.where(is_ok, tl.maximum(m_i, s_msk), m_i)
        was_empty = (m_i == -float("inf"))
        alpha     = tl.where(is_ok, tl.where(was_empty, 0.0, tl.math.exp2(m_i - m_new)), 1.0)
        m_safe    = tl.where(is_ok, m_new, 0.0)
        p_wk      = tl.where(is_ok, tl.math.exp2(s_msk - m_safe), 0.0)

        l_i = l_i * alpha + p_wk
        acc = acc * alpha[:, None] + p_wk[:, None] * v_wk
        m_i = m_new

    y_bf16 = (acc * (1.0 / tl.maximum(l_i, 1e-30))[:, None]).to(tl.bfloat16)
    # Direct write to contiguous `y_out [N_pos * P, W, H, DH]` == `[N_pos * P, W, D]`:
    # `(n_idx * PW + pw_idx) == ((n_idx * P + p_idx) * W + wq_idx)`!
    y_ptrs = y_out_ptr + ((n_idx * PW + pw_idx) * H + pid_h)[:, None] * DH + offs_d[None, :]
    tl.store(y_ptrs, y_bf16, mask=mask_m[:, None])


@triton.jit
def _softcap_selected_bwd_2half_kernel(
    logits_ptr, stride_m, lse_ptr, sum_g_ptr,
    V: tl.constexpr, SOFTCAP: tl.constexpr, INV_SOFTCAP: tl.constexpr, HALF_V: tl.constexpr,
):
    """Single-pass 2-half BF16-register selected-logprobs base backward (`num_warps=8`, 2 waves/SIMD on gfx942)."""
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * stride_m
    lse = tl.load(lse_ptr + row)
    sg  = tl.load(sum_g_ptr + row)
    LOG2E: tl.constexpr = 1.4426950408889634

    offs0 = tl.arange(0, HALF_V)
    offs1 = HALF_V + offs0
    mask1 = offs1 < V

    f0 = tl.load(row_ptr + offs0).to(tl.float32)
    f1 = tl.load(row_ptr + offs1, mask=mask1, other=0.0).to(tl.float32)

    if SOFTCAP > 0.0:
        z0 = tl.minimum(tl.maximum(f0 * INV_SOFTCAP, -20.0), 20.0)
        e0 = tl.math.exp2(2.0 * LOG2E * z0)
        th0 = (e0 - 1.0) / (e0 + 1.0)
        lg0 = SOFTCAP * th0
        ds0 = 1.0 - th0 * th0

        z1 = tl.minimum(tl.maximum(f1 * INV_SOFTCAP, -20.0), 20.0)
        e1 = tl.math.exp2(2.0 * LOG2E * z1)
        th1 = (e1 - 1.0) / (e1 + 1.0)
        lg1 = SOFTCAP * th1
        ds1 = 1.0 - th1 * th1
    else:
        lg0, ds0 = f0, 1.0
        lg1, ds1 = f1, 1.0

    g0 = (-sg * tl.math.exp2((lg0 - lse) * LOG2E)) * ds0
    g1 = (-sg * tl.math.exp2((lg1 - lse) * LOG2E)) * ds1
    tl.store(row_ptr + offs0, g0.to(tl.bfloat16))
    tl.store(row_ptr + offs1, g1.to(tl.bfloat16), mask=mask1)


@torch.no_grad()
def advance_local_window_kv_ultra(
    blocks: nn.ModuleList,
    kv_list: list[tuple[Tensor, Tensor]],
    h_win: Tensor,
    positions: Tensor,
    cos_full: Tensor,
    sin_full: Tensor,
    kv_slot: Tensor,
    block_t: int = 64,
    num_warps: int = 4,
    waves_per_eu: int = 2,
) -> Tensor:
    """
    Zero-intermediate-buffer Probe Transformer Block loop:
    Each block executes ONLY 4 dense hipBLASLt GEMMs + 4 fused Triton kernels!
    """
    N_pos, P, W, D = h_win.shape
    B_u = kv_list[0][0].shape[0]
    H, Dh = blocks[0].cfg.n_heads, blocks[0].cfg.head_dim
    inv_sqrt_dh = 1.0 / math.sqrt(float(Dh))
    PW = P * W
    tot_pw = N_pos * PW

    pos_i32  = positions.to(torch.int32).contiguous()
    slot_i32 = kv_slot.to(torch.int32).contiguous()
    G = max(1, 128 // PW)
    block_m = 128 if PW <= 128 else triton.next_power_of_2(PW)
    grp_starts, grp_ends = build_seq_groups(slot_i32, B_u, G)
    n_grps = grp_starts.shape[0]

    T_max = cos_full.shape[2]
    cos_c = cos_full.reshape(T_max, Dh // 2).contiguous()
    sin_c = sin_full.reshape(T_max, Dh // 2).contiguous()

    # Reusable `[N_pos, H, PW, Dh]` BF16 buffers (only 1.6 GB total at N_pos=8192, PW=40, reused across all blocks!)
    q_pw    = torch.empty((N_pos, H, PW, Dh), dtype=torch.bfloat16, device=h_win.device)
    k_pw    = torch.empty((N_pos, H, PW, Dh), dtype=torch.bfloat16, device=h_win.device)
    v_pw    = torch.empty((N_pos, H, PW, Dh), dtype=torch.bfloat16, device=h_win.device)
    y_attn  = torch.empty((N_pos * P, W, D), dtype=torch.bfloat16, device=h_win.device)

    x_flat = h_win.reshape(N_pos * P, W, D)
    for blk, (k_clean, v_clean) in zip(blocks, kv_list):
        xn  = FusedRMSNormFn.apply(x_flat, blk.attn_norm.weight, blk.attn_norm.eps)
        qkv = blk.wqkv(xn)  # contiguous [N_pos * P, W, 3 * D] == [N_pos * PW, 3, H, Dh]

        _probe_qkv_norm_rope_nhpwd_kernel[(tot_pw,)](
            qkv, pos_i32, blk.q_norm.weight, blk.k_norm.weight, cos_c, sin_c,
            q_pw, k_pw, v_pw,
            N_POS=N_pos, PW=PW, W=W, H=H, DH=Dh, HALF=Dh // 2, EPS=float(blk.q_norm.eps), num_warps=4,
        )

        extra_kw = {"waves_per_eu": waves_per_eu} if waves_per_eu > 0 else {}
        _grouped_flash_prefix_and_local_kernel[(n_grps, H)](
            q_pw, k_clean, v_clean, k_pw, v_pw,
            grp_starts, grp_ends, slot_i32, pos_i32,
            y_attn,
            q_pw.stride(0), q_pw.stride(1), q_pw.stride(2),
            k_clean.stride(0), k_clean.stride(1), k_clean.stride(2),
            SM_SCALE=inv_sqrt_dh, PW=PW, W=W, H=H,
            BLOCK_M=block_m, BLOCK_T=block_t, DH=64,
            num_warps=num_warps, **extra_kw,
        )

        x_flat = x_flat + blk.wo(y_attn)
        mn = FusedRMSNormFn.apply(x_flat, blk.mlp_norm.weight, blk.mlp_norm.eps)
        x_flat = x_flat + blk.wdown(FusedSwiGLUFn.apply(blk.wgu(mn)))

    return x_flat.view(N_pos, P, W, D)
