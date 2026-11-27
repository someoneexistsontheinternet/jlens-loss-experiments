"""
MI325X (gfx942 CDNA3) High-Throughput Block-Scaled FP8 Linear Kernels:
  1. `bf16`               : Standard BF16 F.linear (`hipBLASLt` baseline).
  2. `fp8_rowwise`        : Fused Triton rowwise + fused 2D transpose-rowwise FP8 (`float8_e4m3fnuz`, max=240.0) + `hipBLASLt` `torch._scaled_mm`.
  3. `mxfp8_e8m0`         : Block-scaled FP8 + `UE8M0` power-of-2 scales (`1 x BLOCK_K`, BLOCK_K in {32, 64, 128})
                            with configurable `e8m0_mode` ("ceil_240", "ceil_pow2", "rne_240", "sr_240"),
                            fused 2D transpose-quantize for dgrad/wgrad, L2-swizzled CDNA3 MFMA autotuning, and deterministic Split-K wgrad!
  4. `fp8_hier_bf16_fp32` : NVFP4-inspired Two-Level FP8 — fine `BF16` per-block (`1 x BLOCK_K` or `128 x 128` 2D weight tile) scales
                            inside the MFMA K-loop + outer `FP32` per-channel scales folded into the GEMM epilogue at zero inner-loop cost!
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def detect_fp8_spec():
    """MI300X / MI325X (gfx942) uses FNUZ FP8 (`max = 240.0`); CDNA4 (gfx950) & CUDA use OCP E4M3FN (`max = 448.0`)."""
    if torch.cuda.is_available() and getattr(torch.version, "hip", None) is not None:
        arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
        if "gfx94" in arch:
            return torch.float8_e4m3fnuz, tl.float8e4b8, 240.0, True
    return torch.float8_e4m3fn, tl.float8e4nv, 448.0, False


TORCH_FP8_DTYPE, TL_FP8_DTYPE, FP8_MAX, IS_GFX942 = detect_fp8_spec()

E8M0_MODE_MAP = {
    "ceil_240":  0,  # Tight zero-clipping power-of-2 scale against exact FP8_MAX (240.0 on gfx942)
    "ceil_pow2": 1,  # Power-of-2 binade floor(log2(FP8_MAX)) (128.0 on gfx942)
    "rne_240":   2,  # Round-to-nearest exponent log2(amax / FP8_MAX) with saturating clamp
    "sr_240":    3,  # Stochastic rounding on exponent log2(amax / FP8_MAX) with saturating clamp
}


# ==============================================================================
# 1A. Fused Rowwise & Fused 2D Transpose-Rowwise FP8 Quantizers (for `fp8_rowwise`)
# ==============================================================================

@triton.jit
def _quant_rowwise_fp8_kernel(
    x_ptr, q_ptr, s_ptr, M, K,
    FP8_MAX_VAL: tl.constexpr, INV_FP8_MAX: tl.constexpr,
    BLOCK_K: tl.constexpr, FP8_TY: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    x = tl.load(x_ptr + row * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-12)
    s = amax * INV_FP8_MAX
    scaled = tl.minimum(tl.maximum(x / s, -FP8_MAX_VAL), FP8_MAX_VAL)
    scaled = tl.where(scaled == 0.0, 0.0, scaled)
    tl.store(q_ptr + row * K + offs_k, scaled.to(FP8_TY), mask=mask_k)
    tl.store(s_ptr + row, s)


@triton.jit
def _col_amax_partial_kernel(
    x_ptr, part_ptr, R, C,
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid_r = tl.program_id(0).to(tl.int64)
    pid_c = tl.program_id(1).to(tl.int64)
    n_c_blks = tl.cdiv(C, BLOCK_C)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (offs_r[:, None] < R) & (offs_c[None, :] < C)
    x = tl.load(x_ptr + offs_r[:, None] * C + offs_c[None, :], mask=mask, other=0.0).to(tl.float32)
    cmax = tl.max(tl.abs(x), axis=0)
    tl.store(part_ptr + pid_r * C + offs_c, cmax, mask=(offs_c < C))


@triton.jit
def _transpose_scale_fp8_2d_kernel(
    x_ptr, qt_ptr, s_col_ptr, R, C,
    FP8_MAX_VAL: tl.constexpr,
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr, FP8_TY: tl.constexpr,
):
    pid_r = tl.program_id(0).to(tl.int64)
    pid_c = tl.program_id(1).to(tl.int64)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_in = (offs_r[:, None] < R) & (offs_c[None, :] < C)
    x = tl.load(x_ptr + offs_r[:, None] * C + offs_c[None, :], mask=mask_in, other=0.0).to(tl.float32)
    sc = tl.load(s_col_ptr + offs_c, mask=(offs_c < C), other=1.0).to(tl.float32)
    scaled = tl.minimum(tl.maximum(x / sc[None, :], -FP8_MAX_VAL), FP8_MAX_VAL)
    scaled = tl.where(scaled == 0.0, 0.0, scaled)
    qt_tile = tl.trans(scaled.to(FP8_TY))
    mask_out = (offs_c[:, None] < C) & (offs_r[None, :] < R)
    tl.store(qt_ptr + offs_c[:, None] * R + offs_r[None, :], qt_tile, mask=mask_out)


def quantize_rowwise_fp8(x_2d: torch.Tensor):
    M, K = x_2d.shape
    x_c = x_2d if x_2d.is_contiguous() else x_2d.contiguous()
    if K <= 8192:
        q = torch.empty((M, K), dtype=TORCH_FP8_DTYPE, device=x_c.device)
        s = torch.empty((M, 1), dtype=torch.float32, device=x_c.device)
        bk = max(64, triton.next_power_of_2(K))
        nw = 8 if bk >= 2048 else 4
        _quant_rowwise_fp8_kernel[(M,)](
            x_c, q, s, M, K, FP8_MAX_VAL=FP8_MAX, INV_FP8_MAX=1.0 / FP8_MAX,
            BLOCK_K=bk, FP8_TY=TL_FP8_DTYPE, num_warps=nw,
        )
        return q, s
    s = (x_c.abs().amax(dim=-1, keepdim=True).float().clamp_min(1e-12) * (1.0 / FP8_MAX)).contiguous()
    q = (x_c.float() / s).clamp(-FP8_MAX, FP8_MAX).to(TORCH_FP8_DTYPE)
    return q, s


def quantize_transpose_rowwise_fp8(x_rc: torch.Tensor):
    """Given X of shape (R, C), returns contiguous Q_T of shape (C, R) in FP8 and row-scale S of shape (C, 1) in FP32."""
    R, C = x_rc.shape
    x_c = x_rc if x_rc.is_contiguous() else x_rc.contiguous()
    BR_MAX = 256 if R >= 4096 else 64
    BC_MAX = 64
    n_rb = triton.cdiv(R, BR_MAX)
    part = torch.empty((n_rb, C), dtype=torch.float32, device=x_c.device)
    _col_amax_partial_kernel[(n_rb, triton.cdiv(C, BC_MAX))](
        x_c, part, R, C, BLOCK_R=BR_MAX, BLOCK_C=BC_MAX, num_warps=4
    )
    s_col = (part.amax(dim=0).clamp_min(1e-12) * (1.0 / FP8_MAX)).contiguous()
    qt = torch.empty((C, R), dtype=TORCH_FP8_DTYPE, device=x_c.device)
    BT = 64
    _transpose_scale_fp8_2d_kernel[(triton.cdiv(R, BT), triton.cdiv(C, BT))](
        x_c, qt, s_col, R, C, FP8_MAX_VAL=FP8_MAX, BLOCK_R=BT, BLOCK_C=BT, FP8_TY=TL_FP8_DTYPE, num_warps=4
    )
    return qt, s_col.view(C, 1)


# ==============================================================================
# 1B. MXFP8 `UE8M0` Quantizer + Fused 2D Transpose-Quantizer
# ==============================================================================

@triton.jit
def _compute_e8m0_exp(amax, seed, rng_offs, E8M0_MODE: tl.constexpr, FP8_MAX_POW2_EXP: tl.constexpr, INV_FP8_MAX: tl.constexpr):
    if E8M0_MODE == 1:
        e_unb = tl.math.ceil(tl.math.log2(amax)) - FP8_MAX_POW2_EXP
    elif E8M0_MODE == 2:
        e_unb = tl.math.floor(tl.math.log2(amax * INV_FP8_MAX) + 0.5)
    elif E8M0_MODE == 3:
        u = tl.rand(seed, rng_offs)
        e_unb = tl.math.floor(tl.math.log2(amax * INV_FP8_MAX) + u)
    else:
        e_unb = tl.math.ceil(tl.math.log2(amax * INV_FP8_MAX))
    return tl.minimum(tl.maximum(e_unb, -126.0), 126.0)


@triton.jit
def _quant_mxfp8_e8m0_kernel(
    x_ptr, q_ptr, exp_ptr, M, K, seed,
    E8M0_MODE: tl.constexpr,
    FP8_MAX_VAL: tl.constexpr, INV_FP8_MAX: tl.constexpr, FP8_MAX_POW2_EXP: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, FP8_TY: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1).to(tl.int64)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask   = mask_m[:, None] & (offs_k[None, :] < K)

    x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-30)
    rng_offs = (pid_m * 131071 + pid_k * 8191 + tl.arange(0, BLOCK_M)).to(tl.int32)
    e_clamped = _compute_e8m0_exp(amax, seed, rng_offs, E8M0_MODE, FP8_MAX_POW2_EXP, INV_FP8_MAX)
    inv_scale = tl.math.exp2(-e_clamped)
    scaled = tl.minimum(tl.maximum(x * inv_scale[:, None], -FP8_MAX_VAL), FP8_MAX_VAL)
    scaled = tl.where(scaled == 0.0, 0.0, scaled)

    n_k_blks = K // BLOCK_K
    tl.store(q_ptr + offs_m[:, None] * K + offs_k[None, :], scaled.to(FP8_TY), mask=mask)
    tl.store(exp_ptr + offs_m * n_k_blks + pid_k, (e_clamped + 127.0).to(tl.int32).to(tl.uint8), mask=mask_m)


@triton.jit
def _quant_transpose_mxfp8_e8m0_kernel(
    x_ptr, qt_ptr, exp_t_ptr, R, C, seed,
    E8M0_MODE: tl.constexpr,
    FP8_MAX_VAL: tl.constexpr, INV_FP8_MAX: tl.constexpr, FP8_MAX_POW2_EXP: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr, FP8_TY: tl.constexpr,
):
    """Fused 2D Tile Transpose + MXFP8 Quantizer: X(R, C) -> Q_T(C, R) + E_T(C, R // BLOCK_K)."""
    pid_rb = tl.program_id(0).to(tl.int64)
    pid_cb = tl.program_id(1).to(tl.int64)
    offs_r = pid_rb * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_c = pid_cb * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    mask_in = (offs_r[:, None] < R) & mask_c[None, :]

    x_tile = tl.load(x_ptr + offs_r[:, None] * C + offs_c[None, :], mask=mask_in, other=0.0).to(tl.float32)
    xt = tl.trans(x_tile)  # (BLOCK_C, BLOCK_K)
    amax = tl.maximum(tl.max(tl.abs(xt), axis=1), 1e-30)
    rng_offs = (pid_cb * 131071 + pid_rb * 8191 + tl.arange(0, BLOCK_C)).to(tl.int32)
    e_clamped = _compute_e8m0_exp(amax, seed, rng_offs, E8M0_MODE, FP8_MAX_POW2_EXP, INV_FP8_MAX)
    inv_scale = tl.math.exp2(-e_clamped)
    scaled = tl.minimum(tl.maximum(xt * inv_scale[:, None], -FP8_MAX_VAL), FP8_MAX_VAL)
    scaled = tl.where(scaled == 0.0, 0.0, scaled)

    n_r_blks = R // BLOCK_K
    mask_out = mask_c[:, None] & (offs_r[None, :] < R)
    tl.store(qt_ptr + offs_c[:, None] * R + offs_r[None, :], scaled.to(FP8_TY), mask=mask_out)
    tl.store(exp_t_ptr + offs_c * n_r_blks + pid_rb, (e_clamped + 127.0).to(tl.int32).to(tl.uint8), mask=mask_c)


_SR_STEP_SEED = 1337


def quantize_mxfp8_e8m0(x_2d: torch.Tensor, block_k: int = 64, e8m0_mode: str = "ceil_240"):
    global _SR_STEP_SEED
    _SR_STEP_SEED = (_SR_STEP_SEED + 1) & 0x7FFFFFFF
    M, K = x_2d.shape
    assert K % block_k == 0, f"K={K} must be divisible by block_k={block_k}"
    x_c = x_2d if x_2d.is_contiguous() else x_2d.contiguous()
    q   = torch.empty((M, K), dtype=TORCH_FP8_DTYPE, device=x_c.device)
    exp = torch.empty((M, K // block_k), dtype=torch.uint8, device=x_c.device)
    max_p2 = 7 if FP8_MAX < 256.0 else 8
    emode = E8M0_MODE_MAP.get(e8m0_mode, 0)
    BM = 32 if M >= 32 else (16 if M >= 16 else 1)
    _quant_mxfp8_e8m0_kernel[(triton.cdiv(M, BM), K // block_k)](
        x_c, q, exp, M, K, _SR_STEP_SEED,
        E8M0_MODE=emode, FP8_MAX_VAL=FP8_MAX, INV_FP8_MAX=1.0 / FP8_MAX, FP8_MAX_POW2_EXP=max_p2,
        BLOCK_M=BM, BLOCK_K=block_k, FP8_TY=TL_FP8_DTYPE, num_warps=4,
    )
    return q, exp


def quantize_transpose_mxfp8_e8m0(x_rc: torch.Tensor, block_k: int = 64, e8m0_mode: str = "ceil_240"):
    global _SR_STEP_SEED
    _SR_STEP_SEED = (_SR_STEP_SEED + 1) & 0x7FFFFFFF
    R, C = x_rc.shape
    assert R % block_k == 0, f"R={R} must be divisible by block_k={block_k}"
    x_c = x_rc if x_rc.is_contiguous() else x_rc.contiguous()
    qt   = torch.empty((C, R), dtype=TORCH_FP8_DTYPE, device=x_c.device)
    expt = torch.empty((C, R // block_k), dtype=torch.uint8, device=x_c.device)
    max_p2 = 7 if FP8_MAX < 256.0 else 8
    emode = E8M0_MODE_MAP.get(e8m0_mode, 0)
    BC = 64 if block_k <= 64 else 32
    _quant_transpose_mxfp8_e8m0_kernel[(R // block_k, triton.cdiv(C, BC))](
        x_c, qt, expt, R, C, _SR_STEP_SEED,
        E8M0_MODE=emode, FP8_MAX_VAL=FP8_MAX, INV_FP8_MAX=1.0 / FP8_MAX, FP8_MAX_POW2_EXP=max_p2,
        BLOCK_K=block_k, BLOCK_C=BC, FP8_TY=TL_FP8_DTYPE, num_warps=4,
    )
    return qt, expt


# ==============================================================================
# 1C. Two-Level Hierarchical `BF16-Block + FP32-Channel` FP8 Quantizer + Fused Transpose
# ==============================================================================

@triton.jit
def _quant_hier_row_1pass_kernel(
    x_ptr, q_ptr, s_blk_ptr, s_row_ptr, M, K,
    FP8_MAX_VAL: tl.constexpr, INV_FP8_MAX: tl.constexpr,
    N_BLKS: tl.constexpr, BLOCK_K: tl.constexpr, FP8_TY: tl.constexpr,
):
    """Single-kernel rowwise + fine-block quantizer for K <= 4096 (all 70M fwd & dgrad shapes!)."""
    row = tl.program_id(0).to(tl.int64)
    offs_2d = tl.arange(0, N_BLKS)[:, None] * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
    mask_2d = offs_2d < K
    x = tl.load(x_ptr + row * K + offs_2d, mask=mask_2d, other=0.0).to(tl.float32)

    blk_amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-12)  # (N_BLKS,)
    row_amax = tl.maximum(tl.max(blk_amax, axis=0), 1e-12)
    s_r = row_amax * INV_FP8_MAX
    rel_bf16 = ((blk_amax * INV_FP8_MAX) / tl.maximum(s_r, 1e-30)).to(tl.bfloat16)
    eff_s = tl.maximum(s_r * rel_bf16.to(tl.float32), 1e-30)

    scaled = tl.minimum(tl.maximum(x / eff_s[:, None], -FP8_MAX_VAL), FP8_MAX_VAL)
    scaled = tl.where(scaled == 0.0, 0.0, scaled)

    n_valid_blks = K // BLOCK_K
    blk_ids = tl.arange(0, N_BLKS)
    tl.store(q_ptr + row * K + offs_2d, scaled.to(FP8_TY), mask=mask_2d)
    tl.store(s_blk_ptr + row * n_valid_blks + blk_ids, rel_bf16, mask=(blk_ids < n_valid_blks))
    tl.store(s_row_ptr + row, s_r)


@triton.jit
def _quant_hier_fp8_kernel(
    x_ptr, q_ptr, s_blk_ptr, s_row_ptr, M, K,
    FP8_MAX_VAL: tl.constexpr, INV_FP8_MAX: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, FP8_TY: tl.constexpr,
):
    pid_m  = tl.program_id(0).to(tl.int64)
    pid_k  = tl.program_id(1).to(tl.int64)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask   = mask_m[:, None] & (offs_k[None, :] < K)

    x    = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask, other=0.0).to(tl.float32)
    s_r  = tl.load(s_row_ptr + offs_m, mask=mask_m, other=1.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-12)

    raw_blk_s = amax * INV_FP8_MAX
    rel_bf16  = (raw_blk_s / tl.maximum(s_r, 1e-30)).to(tl.bfloat16)
    eff_s     = tl.maximum(s_r * rel_bf16.to(tl.float32), 1e-30)
    scaled    = tl.minimum(tl.maximum(x / eff_s[:, None], -FP8_MAX_VAL), FP8_MAX_VAL)
    scaled    = tl.where(scaled == 0.0, 0.0, scaled)

    n_k_blks = K // BLOCK_K
    tl.store(q_ptr + offs_m[:, None] * K + offs_k[None, :], scaled.to(FP8_TY), mask=mask)
    tl.store(s_blk_ptr + offs_m * n_k_blks + pid_k, rel_bf16, mask=mask_m)


@triton.jit
def _quant_transpose_hier_fp8_kernel(
    x_ptr, qt_ptr, s_blk_t_ptr, s_col_ptr, R, C,
    FP8_MAX_VAL: tl.constexpr, INV_FP8_MAX: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr, FP8_TY: tl.constexpr,
):
    """Fused 2D Tile Transpose + Hierarchical BF16/FP32 Quantizer: X(R, C) -> Q_T(C, R) + S_blk_T(C, R//BLOCK_K)."""
    pid_rb = tl.program_id(0).to(tl.int64)
    pid_cb = tl.program_id(1).to(tl.int64)
    offs_r = pid_rb * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_c = pid_cb * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    mask_in = (offs_r[:, None] < R) & mask_c[None, :]

    x_tile = tl.load(x_ptr + offs_r[:, None] * C + offs_c[None, :], mask=mask_in, other=0.0).to(tl.float32)
    xt = tl.trans(x_tile)  # (BLOCK_C, BLOCK_K)
    s_c = tl.load(s_col_ptr + offs_c, mask=mask_c, other=1.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(xt), axis=1), 1e-12)

    rel_bf16 = ((amax * INV_FP8_MAX) / tl.maximum(s_c, 1e-30)).to(tl.bfloat16)
    eff_s = tl.maximum(s_c * rel_bf16.to(tl.float32), 1e-30)
    scaled = tl.minimum(tl.maximum(xt / eff_s[:, None], -FP8_MAX_VAL), FP8_MAX_VAL)
    scaled = tl.where(scaled == 0.0, 0.0, scaled)

    n_r_blks = R // BLOCK_K
    mask_out = mask_c[:, None] & (offs_r[None, :] < R)
    tl.store(qt_ptr + offs_c[:, None] * R + offs_r[None, :], scaled.to(FP8_TY), mask=mask_out)
    tl.store(s_blk_t_ptr + offs_c * n_r_blks + pid_rb, rel_bf16, mask=mask_c)


@triton.jit
def _quant_hier_2d_tile_weight_kernel(
    w_ptr, q_ptr, s_tile_ptr, s_row_ptr, N, K,
    FP8_MAX_VAL: tl.constexpr, INV_FP8_MAX: tl.constexpr,
    TILE_N: tl.constexpr, TILE_K: tl.constexpr, FP8_TY: tl.constexpr,
):
    """2D (128x128 or TILE_N x TILE_K) BF16 block scale + outer FP32 per-channel scale for weights."""
    pid_n = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1).to(tl.int64)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = pid_k * TILE_K + tl.arange(0, TILE_K)
    mask_n = offs_n < N
    mask   = mask_n[:, None] & (offs_k[None, :] < K)

    w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], mask=mask, other=0.0).to(tl.float32)
    s_r = tl.load(s_row_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
    normed_abs = tl.abs(w) * (INV_FP8_MAX / tl.maximum(s_r[:, None], 1e-30))
    tile_max = tl.max(tl.max(normed_abs, axis=1), axis=0)
    tile_rel = tl.maximum(tile_max, 1e-7).to(tl.bfloat16)
    eff_s = tl.maximum(s_r * tile_rel.to(tl.float32), 1e-30)

    scaled = tl.minimum(tl.maximum(w / eff_s[:, None], -FP8_MAX_VAL), FP8_MAX_VAL)
    scaled = tl.where(scaled == 0.0, 0.0, scaled)
    tl.store(q_ptr + offs_n[:, None] * K + offs_k[None, :], scaled.to(FP8_TY), mask=mask)
    n_k_blks = K // TILE_K
    tile_vec = (tl.zeros((TILE_N,), dtype=tl.float32) + tile_rel.to(tl.float32)).to(tl.bfloat16)
    tl.store(s_tile_ptr + offs_n * n_k_blks + pid_k, tile_vec, mask=mask_n)


def quantize_hier_bf16_fp32(x_2d: torch.Tensor, block_k: int = 128, tile_2d: bool = False):
    M, K = x_2d.shape
    assert K % block_k == 0, f"K={K} must be divisible by block_k={block_k}"
    x_c = x_2d if x_2d.is_contiguous() else x_2d.contiguous()
    q     = torch.empty((M, K), dtype=TORCH_FP8_DTYPE, device=x_c.device)
    s_blk = torch.empty((M, K // block_k), dtype=torch.bfloat16, device=x_c.device)

    if tile_2d and M % 128 == 0:
        s_row = (x_c.abs().amax(dim=-1).float().clamp_min(1e-12) * (1.0 / FP8_MAX)).contiguous()
        _quant_hier_2d_tile_weight_kernel[(M // 128, K // block_k)](
            x_c, q, s_blk, s_row, M, K,
            FP8_MAX_VAL=FP8_MAX, INV_FP8_MAX=1.0 / FP8_MAX,
            TILE_N=128, TILE_K=block_k, FP8_TY=TL_FP8_DTYPE, num_warps=8,
        )
        return q, s_blk, s_row

    if K <= 4096:
        s_row = torch.empty((M,), dtype=torch.float32, device=x_c.device)
        n_blks_p2 = max(1, triton.next_power_of_2(K // block_k))
        nw = 8 if K >= 2048 else 4
        _quant_hier_row_1pass_kernel[(M,)](
            x_c, q, s_blk, s_row, M, K,
            FP8_MAX_VAL=FP8_MAX, INV_FP8_MAX=1.0 / FP8_MAX,
            N_BLKS=n_blks_p2, BLOCK_K=block_k, FP8_TY=TL_FP8_DTYPE, num_warps=nw,
        )
        return q, s_blk, s_row

    s_row = (x_c.abs().amax(dim=-1).float().clamp_min(1e-12) * (1.0 / FP8_MAX)).contiguous()
    BM = 32 if M >= 32 else 16
    _quant_hier_fp8_kernel[(triton.cdiv(M, BM), K // block_k)](
        x_c, q, s_blk, s_row, M, K,
        FP8_MAX_VAL=FP8_MAX, INV_FP8_MAX=1.0 / FP8_MAX,
        BLOCK_M=BM, BLOCK_K=block_k, FP8_TY=TL_FP8_DTYPE, num_warps=4,
    )
    return q, s_blk, s_row


def quantize_transpose_hier_bf16_fp32(x_rc: torch.Tensor, block_k: int = 128):
    R, C = x_rc.shape
    assert R % block_k == 0, f"R={R} must be divisible by block_k={block_k}"
    x_c = x_rc if x_rc.is_contiguous() else x_rc.contiguous()
    BR_MAX = 256 if R >= 4096 else 64
    BC_MAX = 64
    n_rb = triton.cdiv(R, BR_MAX)
    part = torch.empty((n_rb, C), dtype=torch.float32, device=x_c.device)
    _col_amax_partial_kernel[(n_rb, triton.cdiv(C, BC_MAX))](
        x_c, part, R, C, BLOCK_R=BR_MAX, BLOCK_C=BC_MAX, num_warps=4
    )
    s_col = (part.amax(dim=0).clamp_min(1e-12) * (1.0 / FP8_MAX)).contiguous()
    qt = torch.empty((C, R), dtype=TORCH_FP8_DTYPE, device=x_c.device)
    s_blk_t = torch.empty((C, R // block_k), dtype=torch.bfloat16, device=x_c.device)
    BC = 64 if block_k <= 64 else 32
    _quant_transpose_hier_fp8_kernel[(R // block_k, triton.cdiv(C, BC))](
        x_c, qt, s_blk_t, s_col, R, C,
        FP8_MAX_VAL=FP8_MAX, INV_FP8_MAX=1.0 / FP8_MAX,
        BLOCK_K=block_k, BLOCK_C=BC, FP8_TY=TL_FP8_DTYPE, num_warps=4,
    )
    return qt, s_blk_t, s_col


# ==============================================================================
# 2. CDNA3 (gfx942) MFMA Autotuned + Deterministic Split-K Block-Scaled FP8 GEMMs
# ==============================================================================

def _make_rocm_cfg(kw: dict, num_warps: int, num_stages: int, waves_per_eu: int, matrix_instr_nonkdim: int):
    try:
        return triton.Config(kw, num_warps=num_warps, num_stages=num_stages,
                             waves_per_eu=waves_per_eu, matrix_instr_nonkdim=matrix_instr_nonkdim)
    except TypeError:
        return triton.Config(kw, num_warps=num_warps, num_stages=num_stages)


def _build_gemm_configs():
    if IS_GFX942:
        return [
            _make_rocm_cfg({"BLOCK_M": 256, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=4, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16),
            _make_rocm_cfg({"BLOCK_M": 128, "BLOCK_N": 256, "GROUP_M": 8}, num_warps=4, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16),
            _make_rocm_cfg({"BLOCK_M": 256, "BLOCK_N": 256, "GROUP_M": 4}, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=16),
            _make_rocm_cfg({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=4, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16),
            _make_rocm_cfg({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),
            _make_rocm_cfg({"BLOCK_M": 64,  "BLOCK_N": 128, "GROUP_M": 8}, num_warps=4, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16),
            _make_rocm_cfg({"BLOCK_M": 128, "BLOCK_N": 64,  "GROUP_M": 8}, num_warps=4, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16),
        ]
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "GROUP_M": 8}, num_warps=4, num_stages=2),
    ]


_GEMM_CONFIGS = _build_gemm_configs()


def _select_split_k(M: int, N: int, K: int, block_k: int) -> int:
    n_k_blks = K // block_k
    mn_tiles = triton.cdiv(M, 128) * triton.cdiv(N, 128)
    if mn_tiles <= 32 and n_k_blks >= 64 and n_k_blks % 32 == 0:
        return 32
    if mn_tiles <= 128 and n_k_blks >= 32 and n_k_blks % 16 == 0:
        return 16
    if mn_tiles <= 256 and n_k_blks >= 16 and n_k_blks % 8 == 0:
        return 8
    return 1


@triton.jit
def _swizzle_pid(pid, num_pid_m, num_pid_n, GROUP_M: tl.constexpr):
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m.to(tl.int64), pid_n.to(tl.int64)


@triton.autotune(configs=_GEMM_CONFIGS, key=["M", "N", "K", "SCALE_K", "SPLIT_K"])
@triton.jit
def _mxfp8_e8m0_gemm_nt_kernel(
    a_ptr, b_ptr, ea_ptr, eb_ptr, c_ptr,
    M, N, K,
    SCALE_K: tl.constexpr, SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_sk = tl.program_id(1).to(tl.int64)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = _swizzle_pid(pid, num_pid_m, num_pid_n, GROUP_M)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, SCALE_K).to(tl.int64)
    n_k_blks = K // SCALE_K
    blks_per_sk = n_k_blks // SPLIT_K
    kb_start = pid_sk * blks_per_sk
    kb_end   = kb_start + blks_per_sk

    K_i64 = pid_sk * 0 + K
    N_i64 = pid_sk * 0 + N
    M_i64 = pid_sk * 0 + M
    mask_m = offs_m < M_i64
    mask_n = offs_n < N_i64

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(kb_start, kb_end):
        k_cur = kb * SCALE_K + offs_k
        a_tile = tl.load(a_ptr + offs_m[:, None] * K_i64 + k_cur[None, :], mask=mask_m[:, None], other=0.0)
        b_tile = tl.load(b_ptr + offs_n[:, None] * K_i64 + k_cur[None, :], mask=mask_n[:, None], other=0.0)
        ea = tl.load(ea_ptr + offs_m * n_k_blks + kb, mask=mask_m, other=127).to(tl.int32)
        eb = tl.load(eb_ptr + offs_n * n_k_blks + kb, mask=mask_n, other=127).to(tl.int32)

        tile_dot = tl.dot(a_tile, tl.trans(b_tile))
        sa = tl.math.exp2((ea - 127).to(tl.float32))
        sb = tl.math.exp2((eb - 127).to(tl.float32))
        acc += tile_dot * (sa[:, None] * sb[None, :])

    mask_c = mask_m[:, None] & mask_n[None, :]
    out_off = pid_sk * (M_i64 * N_i64) + offs_m[:, None] * N_i64 + offs_n[None, :]
    tl.store(c_ptr + out_off, acc.to(c_ptr.dtype.element_ty), mask=mask_c)


@triton.autotune(configs=_GEMM_CONFIGS, key=["M", "N", "K", "SCALE_K", "SPLIT_K"])
@triton.jit
def _hier_bf16_fp32_gemm_nt_kernel(
    a_ptr, b_ptr, sa_blk_ptr, sb_blk_ptr, sa_row_ptr, sb_col_ptr, c_ptr,
    M, N, K,
    SCALE_K: tl.constexpr, SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_sk = tl.program_id(1).to(tl.int64)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = _swizzle_pid(pid, num_pid_m, num_pid_n, GROUP_M)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, SCALE_K).to(tl.int64)
    n_k_blks = K // SCALE_K
    blks_per_sk = n_k_blks // SPLIT_K
    kb_start = pid_sk * blks_per_sk
    kb_end   = kb_start + blks_per_sk

    K_i64 = pid_sk * 0 + K
    N_i64 = pid_sk * 0 + N
    M_i64 = pid_sk * 0 + M
    mask_m = offs_m < M_i64
    mask_n = offs_n < N_i64

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(kb_start, kb_end):
        k_cur = kb * SCALE_K + offs_k
        a_tile = tl.load(a_ptr + offs_m[:, None] * K_i64 + k_cur[None, :], mask=mask_m[:, None], other=0.0)
        b_tile = tl.load(b_ptr + offs_n[:, None] * K_i64 + k_cur[None, :], mask=mask_n[:, None], other=0.0)
        sa = tl.load(sa_blk_ptr + offs_m * n_k_blks + kb, mask=mask_m, other=0.0).to(tl.float32)
        sb = tl.load(sb_blk_ptr + offs_n * n_k_blks + kb, mask=mask_n, other=0.0).to(tl.float32)

        acc += tl.dot(a_tile, tl.trans(b_tile)) * (sa[:, None] * sb[None, :])

    sr = tl.load(sa_row_ptr + offs_m, mask=mask_m, other=1.0).to(tl.float32)
    sc = tl.load(sb_col_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
    out = acc * (sr[:, None] * sc[None, :])
    mask_c = mask_m[:, None] & mask_n[None, :]
    out_off = pid_sk * (M_i64 * N_i64) + offs_m[:, None] * N_i64 + offs_n[None, :]
    tl.store(c_ptr + out_off, out.to(c_ptr.dtype.element_ty), mask=mask_c)


def _run_mxfp8_gemm_prequant(qa: torch.Tensor, ea: torch.Tensor, qb: torch.Tensor, eb: torch.Tensor, block_k: int, out_dtype):
    M, K = qa.shape
    N = qb.shape[0]
    sk = _select_split_k(M, N, K, block_k)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]), sk)
    if sk == 1:
        out = torch.empty((M, N), dtype=out_dtype, device=qa.device)
        _mxfp8_e8m0_gemm_nt_kernel[grid](qa, qb, ea, eb, out, M, N, K, SCALE_K=block_k, SPLIT_K=1)
        return out
    part = torch.empty((sk, M, N), dtype=torch.float32, device=qa.device)
    _mxfp8_e8m0_gemm_nt_kernel[grid](qa, qb, ea, eb, part, M, N, K, SCALE_K=block_k, SPLIT_K=sk)
    return part.sum(dim=0).to(out_dtype)


def _run_hier_gemm_prequant(qa, sa_b, sa_r, qb, sb_b, sb_c, block_k: int, out_dtype):
    M, K = qa.shape
    N = qb.shape[0]
    sk = _select_split_k(M, N, K, block_k)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]), sk)
    if sk == 1:
        out = torch.empty((M, N), dtype=out_dtype, device=qa.device)
        _hier_bf16_fp32_gemm_nt_kernel[grid](qa, qb, sa_b, sb_b, sa_r, sb_c, out, M, N, K, SCALE_K=block_k, SPLIT_K=1)
        return out
    part = torch.empty((sk, M, N), dtype=torch.float32, device=qa.device)
    _hier_bf16_fp32_gemm_nt_kernel[grid](qa, qb, sa_b, sb_b, sa_r, sb_c, part, M, N, K, SCALE_K=block_k, SPLIT_K=sk)
    return part.sum(dim=0).to(out_dtype)


def gemm_mxfp8_nt(a_2d: torch.Tensor, b_2d: torch.Tensor, block_k: int = 64, e8m0_mode: str = "ceil_240", out_dtype=torch.bfloat16):
    qa, ea = quantize_mxfp8_e8m0(a_2d, block_k=block_k, e8m0_mode=e8m0_mode)
    qb, eb = quantize_mxfp8_e8m0(b_2d, block_k=block_k, e8m0_mode=e8m0_mode)
    return _run_mxfp8_gemm_prequant(qa, ea, qb, eb, block_k, out_dtype)


def gemm_hier_fp8_nt(a_2d: torch.Tensor, b_2d: torch.Tensor, block_k: int = 128, tile_2d_b: bool = False, out_dtype=torch.bfloat16):
    qa, sa_b, sa_r = quantize_hier_bf16_fp32(a_2d, block_k=block_k, tile_2d=False)
    qb, sb_b, sb_c = quantize_hier_bf16_fp32(b_2d, block_k=block_k, tile_2d=tile_2d_b)
    return _run_hier_gemm_prequant(qa, sa_b, sa_r, qb, sb_b, sb_c, block_k, out_dtype)


_ROWWISE_SCALED_MM_OK = None


def _supports_rowwise_scaled_mm(device: torch.device) -> bool:
    global _ROWWISE_SCALED_MM_OK
    if _ROWWISE_SCALED_MM_OK is not None:
        return _ROWWISE_SCALED_MM_OK
    try:
        a = torch.zeros((64, 64), dtype=TORCH_FP8_DTYPE, device=device)
        b = torch.zeros((64, 64), dtype=TORCH_FP8_DTYPE, device=device)
        sa = torch.ones((64, 1), dtype=torch.float32, device=device)
        sb = torch.ones((1, 64), dtype=torch.float32, device=device)
        _ = torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        _ROWWISE_SCALED_MM_OK = True
    except Exception:
        _ROWWISE_SCALED_MM_OK = False
    return _ROWWISE_SCALED_MM_OK


def _scaled_mm_prequant(qa: torch.Tensor, sa_col: torch.Tensor, qb: torch.Tensor, sb_col: torch.Tensor, out_dtype):
    res = torch._scaled_mm(qa, qb.t(), scale_a=sa_col, scale_b=sb_col.t().contiguous(), out_dtype=out_dtype)
    return res[0] if isinstance(res, tuple) else res


def gemm_hipblaslt_fp8_nt(a_2d: torch.Tensor, b_2d: torch.Tensor, out_dtype=torch.bfloat16):
    """Fused Triton rowwise quantization + `hipBLASLt` `torch._scaled_mm`."""
    if _supports_rowwise_scaled_mm(a_2d.device):
        qa, sa = quantize_rowwise_fp8(a_2d)
        qb, sb = quantize_rowwise_fp8(b_2d)
        return _scaled_mm_prequant(qa, sa, qb, sb, out_dtype)
    a_c = a_2d if a_2d.is_contiguous() else a_2d.contiguous()
    b_c = b_2d if b_2d.is_contiguous() else b_2d.contiguous()
    sa = (a_c.abs().amax().float().clamp_min(1e-12) * (1.0 / FP8_MAX)).reshape(())
    sb = (b_c.abs().amax().float().clamp_min(1e-12) * (1.0 / FP8_MAX)).reshape(())
    qa = (a_c.float() / sa).clamp(-FP8_MAX, FP8_MAX).to(TORCH_FP8_DTYPE)
    qb = (b_c.float() / sb).clamp(-FP8_MAX, FP8_MAX).to(TORCH_FP8_DTYPE)
    res = torch._scaled_mm(qa, qb.t(), scale_a=sa, scale_b=sb, out_dtype=out_dtype)
    return res[0] if isinstance(res, tuple) else res


# ==============================================================================
# 3. Unified Autograd Function & Compiler-Friendly `QuantLinear`
# ==============================================================================

class _QuantLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, mode: str, block_k: int, e8m0_mode: str, hier_w_2d: bool):
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
        ctx.save_for_backward(x_2d, w)
        ctx.meta = (orig_shape, mode, block_k, e8m0_mode, hier_w_2d, x.dtype)

        if mode == "mxfp8_e8m0":
            y_2d = gemm_mxfp8_nt(x_2d, w, block_k=block_k, e8m0_mode=e8m0_mode, out_dtype=x.dtype)
        elif mode == "fp8_hier_bf16_fp32":
            y_2d = gemm_hier_fp8_nt(x_2d, w, block_k=block_k, tile_2d_b=hier_w_2d, out_dtype=x.dtype)
        elif mode == "fp8_rowwise":
            y_2d = gemm_hipblaslt_fp8_nt(x_2d, w, out_dtype=x.dtype)
        else:
            y_2d = F.linear(x_2d, w)
        return y_2d.reshape(*orig_shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_2d, w = ctx.saved_tensors
        orig_shape, mode, block_k, e8m0_mode, _hier_w_2d, x_dtype = ctx.meta
        dy_2d = dy.reshape(-1, w.shape[0]).contiguous()

        dx = dw = None
        if ctx.needs_input_grad[0]:
            if mode == "mxfp8_e8m0" and (w.shape[0] % block_k == 0):
                qdy, edy = quantize_mxfp8_e8m0(dy_2d, block_k=block_k, e8m0_mode=e8m0_mode)
                qwt, ewt = quantize_transpose_mxfp8_e8m0(w, block_k=block_k, e8m0_mode=e8m0_mode)
                dx_2d = _run_mxfp8_gemm_prequant(qdy, edy, qwt, ewt, block_k, x_dtype)
            elif mode == "fp8_hier_bf16_fp32" and (w.shape[0] % block_k == 0):
                qdy, sdy_b, sdy_r = quantize_hier_bf16_fp32(dy_2d, block_k=block_k, tile_2d=False)
                qwt, swt_b, swt_c = quantize_transpose_hier_bf16_fp32(w, block_k=block_k)
                dx_2d = _run_hier_gemm_prequant(qdy, sdy_b, sdy_r, qwt, swt_b, swt_c, block_k, x_dtype)
            elif mode == "fp8_rowwise" and _supports_rowwise_scaled_mm(dy_2d.device):
                qdy, sdy = quantize_rowwise_fp8(dy_2d)
                qwt, swt = quantize_transpose_rowwise_fp8(w)
                dx_2d = _scaled_mm_prequant(qdy, sdy, qwt, swt, x_dtype)
            elif mode == "fp8_rowwise":
                dx_2d = gemm_hipblaslt_fp8_nt(dy_2d, w.t().contiguous(), out_dtype=x_dtype)
            else:
                dx_2d = torch.matmul(dy_2d, w)
            dx = dx_2d.reshape(orig_shape)

        if ctx.needs_input_grad[1]:
            M_k = dy_2d.shape[0]
            if mode == "mxfp8_e8m0" and (M_k % block_k == 0):
                qdyt, edyt = quantize_transpose_mxfp8_e8m0(dy_2d, block_k=block_k, e8m0_mode=e8m0_mode)
                qxt, ext   = quantize_transpose_mxfp8_e8m0(x_2d, block_k=block_k, e8m0_mode=e8m0_mode)
                dw = _run_mxfp8_gemm_prequant(qdyt, edyt, qxt, ext, block_k, w.dtype)
            elif mode == "fp8_hier_bf16_fp32" and (M_k % block_k == 0):
                qdyt, sdyt_b, sdyt_r = quantize_transpose_hier_bf16_fp32(dy_2d, block_k=block_k)
                qxt, sxt_b, sxt_c    = quantize_transpose_hier_bf16_fp32(x_2d, block_k=block_k)
                dw = _run_hier_gemm_prequant(qdyt, sdyt_b, sdyt_r, qxt, sxt_b, sxt_c, block_k, w.dtype)
            elif mode == "fp8_rowwise" and _supports_rowwise_scaled_mm(dy_2d.device):
                qdyt, sdyt = quantize_transpose_rowwise_fp8(dy_2d)
                qxt, sxt   = quantize_transpose_rowwise_fp8(x_2d)
                dw = _scaled_mm_prequant(qdyt, sdyt, qxt, sxt, w.dtype)
            elif mode == "fp8_rowwise":
                dw = gemm_hipblaslt_fp8_nt(dy_2d.t().contiguous(), x_2d.t().contiguous(), out_dtype=w.dtype)
            else:
                dw = torch.matmul(dy_2d.t(), x_2d)

        return dx, dw, None, None, None, None


@torch._dynamo.disable
def _quant_linear_dispatch(x: torch.Tensor, w: torch.Tensor, mode: str, block_k: int, e8m0_mode: str, hier_w_2d: bool):
    return _QuantLinearFn.apply(x, w, mode, block_k, e8m0_mode, hier_w_2d)


class QuantLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        mode: str = "bf16",
        block_k: int = 128,
        e8m0_mode: str = "ceil_240",
        hier_w_2d: bool = False,
        std: float = None,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.mode         = mode
        self.block_k      = block_k
        self.e8m0_mode    = e8m0_mode
        self.hier_w_2d    = hier_w_2d
        self.weight       = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.bfloat16))
        init_std = std if std is not None else (1.0 / math.sqrt(in_features))
        nn.init.normal_(self.weight, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "bf16":
            return F.linear(x, self.weight)
        return _quant_linear_dispatch(x, self.weight, self.mode, self.block_k, self.e8m0_mode, self.hier_w_2d)
