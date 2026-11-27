"""
MI325X (gfx942 CDNA3) & Cross-Platform FP8 Linear Kernels:
  1. `bf16`               : Standard BF16 F.linear (`hipBLASLt` / `cuBLAS` baseline).
  2. `fp8_rowwise`        : Rowwise (`1xK` * `Kx1` FP32 scales) or Tensorwise `torch._scaled_mm` in native FP8 (`float8_e4m3fnuz` on MI325X).
  3. `mxfp8_e8m0`         : Block-scaled FP8 + `UE8M0` power-of-2 exponent scales (`1 x BLOCK_K`) accumulated on MFMA FP32 VGPRs via Triton.
  4. `fp8_hier_bf16_fp32` : NVFP4-inspired Two-Level FP8 — fine `BF16` per-block (`1 x BLOCK_K`) scales inside the Triton MFMA K-loop
                            paired with outer `FP32` per-channel / per-tensor scales folded into the GEMM epilogue at zero inner-loop cost!
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


# ==============================================================================
# 1. Fused Quantization Kernels: (A) MXFP8 UE8M0 Power-of-2 & (B) Two-Level BF16-Block + FP32-Channel
# ==============================================================================

@triton.jit
def _quant_mxfp8_e8m0_kernel(
    x_ptr, q_ptr, exp_ptr, M, K,
    FP8_MAX_POW2_EXP: tl.constexpr,  # floor(log2(FP8_MAX)): 7 for 240.0 (2^7=128) or 8 for 448.0 (2^8=256)
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    FP8_TY: tl.constexpr,
):
    """
    Quantizes `x` (M, K) in (1 x BLOCK_K) blocks into FP8 + uint8 UE8M0 biased exponent:
      e_unbiased = ceil(log2(amax)) - FP8_MAX_POW2_EXP  (guarantees |x| / 2^e <= 2^FP8_MAX_POW2_EXP <= FP8_MAX, zero clipping!)
      stored u8  = clamp(e_unbiased + 127, 0, 254)
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask   = mask_m[:, None] & (offs_k[None, :] < K)

    x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-30)
    # Exact power-of-2 ceiling scale via log2:
    e_unb = tl.math.ceil(tl.math.log2(amax)) - FP8_MAX_POW2_EXP
    e_clamped = tl.minimum(tl.maximum(e_unb, -126.0), 126.0)
    inv_scale = tl.math.exp2(-e_clamped)
    xq = (x * inv_scale[:, None]).to(FP8_TY)

    n_k_blks = tl.cdiv(K, BLOCK_K)
    tl.store(q_ptr + offs_m[:, None] * K + offs_k[None, :], xq, mask=mask)
    tl.store(exp_ptr + offs_m * n_k_blks + pid_k, (e_clamped + 127.0).to(tl.int32).to(tl.uint8), mask=mask_m)


def quantize_mxfp8_e8m0(x_2d: torch.Tensor, block_k: int = 64):
    M, K = x_2d.shape
    assert K % block_k == 0, f"K={K} must be divisible by block_k={block_k}"
    x_c = x_2d if x_2d.is_contiguous() else x_2d.contiguous()
    q   = torch.empty((M, K), dtype=TORCH_FP8_DTYPE, device=x_c.device)
    exp = torch.empty((M, K // block_k), dtype=torch.uint8, device=x_c.device)
    max_p2 = 7 if FP8_MAX < 256.0 else 8
    BM = 16 if M >= 16 else 1
    _quant_mxfp8_e8m0_kernel[(triton.cdiv(M, BM), K // block_k)](
        x_c, q, exp, M, K, FP8_MAX_POW2_EXP=max_p2, BLOCK_M=BM, BLOCK_K=block_k, FP8_TY=TL_FP8_DTYPE, num_warps=4
    )
    return q, exp


@triton.jit
def _quant_hier_fp8_kernel(
    x_ptr, q_ptr, s_blk_ptr, s_row_ptr, M, K, INV_FP8_MAX,
    FP8_MAX_VAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, FP8_TY: tl.constexpr,
):
    """
    NVFP4-Inspired Two-Level Hierarchical FP8 Quantizer:
      - Pass-free factorization using precomputed or 1-pass row scale `s_row` (FP32, shape (M,))
        + fine per-(1 x BLOCK_K) block scale `s_blk` stored in BF16 (normalized by s_row so BF16 never overflows/underflows!).
      - Effective real scale of block (m, kb) is: `s_row[m] * float(s_blk[m, kb])`.
    """
    pid_m  = tl.program_id(0)
    pid_k  = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask   = mask_m[:, None] & (offs_k[None, :] < K)

    x    = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask, other=0.0).to(tl.float32)
    s_r  = tl.load(s_row_ptr + offs_m, mask=mask_m, other=1.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-12)

    # Normalized BF16 block multiplier in (0, 1]: exact continuous mantissa (no 2x E8M0 step loss!)
    raw_blk_s = amax * INV_FP8_MAX
    rel_bf16  = (raw_blk_s / tl.maximum(s_r, 1e-30)).to(tl.bfloat16)
    eff_s     = tl.maximum(s_r * rel_bf16.to(tl.float32), 1e-30)
    xq        = tl.minimum(tl.maximum(x / eff_s[:, None], -FP8_MAX_VAL), FP8_MAX_VAL).to(FP8_TY)

    n_k_blks = tl.cdiv(K, BLOCK_K)
    tl.store(q_ptr + offs_m[:, None] * K + offs_k[None, :], xq, mask=mask)
    tl.store(s_blk_ptr + offs_m * n_k_blks + pid_k, rel_bf16, mask=mask_m)


def quantize_hier_bf16_fp32(x_2d: torch.Tensor, block_k: int = 128):
    M, K = x_2d.shape
    assert K % block_k == 0, f"K={K} must be divisible by block_k={block_k}"
    x_c = x_2d if x_2d.is_contiguous() else x_2d.contiguous()
    # Outer FP32 per-channel (row) scale:
    s_row = (x_c.abs().amax(dim=-1).float().clamp_min(1e-12) * (1.0 / FP8_MAX)).contiguous()
    q     = torch.empty((M, K), dtype=TORCH_FP8_DTYPE, device=x_c.device)
    s_blk = torch.empty((M, K // block_k), dtype=torch.bfloat16, device=x_c.device)
    BM = 16 if M >= 16 else 1
    _quant_hier_fp8_kernel[(triton.cdiv(M, BM), K // block_k)](
        x_c, q, s_blk, s_row, M, K, 1.0 / FP8_MAX,
        FP8_MAX_VAL=FP8_MAX, BLOCK_M=BM, BLOCK_K=block_k, FP8_TY=TL_FP8_DTYPE, num_warps=4,
    )
    return q, s_blk, s_row


# ==============================================================================
# 2. ROCm MFMA / Tensor-Core Block-Scaled FP8 GEMM Kernels: C(M, N) = A(M, K) @ B(N, K)^T
# ==============================================================================

_GEMM_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_warps=4, num_stages=2),
]


@triton.autotune(configs=_GEMM_CONFIGS, key=["M", "N", "K", "SCALE_K"])
@triton.jit
def _mxfp8_e8m0_gemm_nt_kernel(
    a_ptr, b_ptr, ea_ptr, eb_ptr, c_ptr,
    M, N, K,
    SCALE_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, SCALE_K)
    n_k_blks = K // SCALE_K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, n_k_blks):
        k_cur = kb * SCALE_K + offs_k
        a_tile = tl.load(a_ptr + offs_m[:, None] * K + k_cur[None, :], mask=(offs_m[:, None] < M), other=0.0)
        b_tile = tl.load(b_ptr + offs_n[:, None] * K + k_cur[None, :], mask=(offs_n[:, None] < N), other=0.0)
        ea = tl.load(ea_ptr + offs_m * n_k_blks + kb, mask=(offs_m < M), other=127).to(tl.int32)
        eb = tl.load(eb_ptr + offs_n * n_k_blks + kb, mask=(offs_n < N), other=127).to(tl.int32)

        # Native FP8 MFMA dot + exact power-of-2 exponent combine in FP32 VGPRs!
        tile_dot = tl.dot(a_tile, tl.trans(b_tile))
        scale_2d = tl.math.exp2((ea[:, None] + eb[None, :] - 254).to(tl.float32))
        acc += tile_dot * scale_2d

    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(c_ptr.dtype.element_ty), mask=mask_c)


@triton.autotune(configs=_GEMM_CONFIGS, key=["M", "N", "K", "SCALE_K"])
@triton.jit
def _hier_bf16_fp32_gemm_nt_kernel(
    a_ptr, b_ptr, sa_blk_ptr, sb_blk_ptr, sa_row_ptr, sb_col_ptr, c_ptr,
    M, N, K,
    SCALE_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Two-Level FP8 GEMM:
      - Inner K-loop: native FP8 MFMA `tl.dot` rescaled by fine BF16 block scales `sa_blk * sb_blk`.
      - Epilogue: multiplied once by outer FP32 per-row & per-channel scales `sa_row * sb_col` (0 inner-loop cost!).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, SCALE_K)
    n_k_blks = K // SCALE_K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, n_k_blks):
        k_cur = kb * SCALE_K + offs_k
        a_tile = tl.load(a_ptr + offs_m[:, None] * K + k_cur[None, :], mask=(offs_m[:, None] < M), other=0.0)
        b_tile = tl.load(b_ptr + offs_n[:, None] * K + k_cur[None, :], mask=(offs_n[:, None] < N), other=0.0)
        sa = tl.load(sa_blk_ptr + offs_m * n_k_blks + kb, mask=(offs_m < M), other=0.0).to(tl.float32)
        sb = tl.load(sb_blk_ptr + offs_n * n_k_blks + kb, mask=(offs_n < N), other=0.0).to(tl.float32)

        acc += tl.dot(a_tile, tl.trans(b_tile)) * (sa[:, None] * sb[None, :])

    sr = tl.load(sa_row_ptr + offs_m, mask=(offs_m < M), other=1.0).to(tl.float32)
    sc = tl.load(sb_col_ptr + offs_n, mask=(offs_n < N), other=1.0).to(tl.float32)
    out = acc * (sr[:, None] * sc[None, :])
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], out.to(c_ptr.dtype.element_ty), mask=mask_c)


def gemm_mxfp8_nt(a_2d: torch.Tensor, b_2d: torch.Tensor, block_k: int = 64, out_dtype=torch.bfloat16):
    M, K = a_2d.shape
    N, Kb = b_2d.shape
    assert K == Kb
    qa, ea = quantize_mxfp8_e8m0(a_2d, block_k=block_k)
    qb, eb = quantize_mxfp8_e8m0(b_2d, block_k=block_k)
    out = torch.empty((M, N), dtype=out_dtype, device=a_2d.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    _mxfp8_e8m0_gemm_nt_kernel[grid](qa, qb, ea, eb, out, M, N, K, SCALE_K=block_k)
    return out


def gemm_hier_fp8_nt(a_2d: torch.Tensor, b_2d: torch.Tensor, block_k: int = 128, out_dtype=torch.bfloat16):
    M, K = a_2d.shape
    N, Kb = b_2d.shape
    assert K == Kb
    qa, sa_b, sa_r = quantize_hier_bf16_fp32(a_2d, block_k=block_k)
    qb, sb_b, sb_c = quantize_hier_bf16_fp32(b_2d, block_k=block_k)
    out = torch.empty((M, N), dtype=out_dtype, device=a_2d.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    _hier_bf16_fp32_gemm_nt_kernel[grid](qa, qb, sa_b, sb_b, sa_r, sb_c, out, M, N, K, SCALE_K=block_k)
    return out


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


def gemm_hipblaslt_fp8_nt(a_2d: torch.Tensor, b_2d: torch.Tensor, out_dtype=torch.bfloat16):
    """Rowwise (with automatic tensorwise fallback) native PyTorch `_scaled_mm` (`hipBLASLt` on MI325X)."""
    a_c = a_2d if a_2d.is_contiguous() else a_2d.contiguous()
    b_c = b_2d if b_2d.is_contiguous() else b_2d.contiguous()
    if _supports_rowwise_scaled_mm(a_c.device):
        sa = (a_c.abs().amax(dim=-1, keepdim=True).float().clamp_min(1e-12) * (1.0 / FP8_MAX)).contiguous()
        sb = (b_c.abs().amax(dim=-1, keepdim=True).float().clamp_min(1e-12) * (1.0 / FP8_MAX)).t().contiguous()
        qa = (a_c.float() / sa).clamp(-FP8_MAX, FP8_MAX).to(TORCH_FP8_DTYPE)
        qb = (b_c.float() / sb.t()).clamp(-FP8_MAX, FP8_MAX).to(TORCH_FP8_DTYPE)
    else:
        sa = (a_c.abs().amax().float().clamp_min(1e-12) * (1.0 / FP8_MAX)).reshape(())
        sb = (b_c.abs().amax().float().clamp_min(1e-12) * (1.0 / FP8_MAX)).reshape(())
        qa = (a_c.float() / sa).clamp(-FP8_MAX, FP8_MAX).to(TORCH_FP8_DTYPE)
        qb = (b_c.float() / sb).clamp(-FP8_MAX, FP8_MAX).to(TORCH_FP8_DTYPE)
    res = torch._scaled_mm(qa, qb.t(), scale_a=sa, scale_b=sb, out_dtype=out_dtype)
    return res[0] if isinstance(res, tuple) else res


# ==============================================================================
# 3. Unified Autograd Function & nn.Module Drop-in (`QuantLinear`)
# ==============================================================================

class _QuantLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, mode: str, block_k: int):
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        ctx.save_for_backward(x_2d, w)
        ctx.meta = (orig_shape, mode, block_k, x.dtype)

        if mode == "mxfp8_e8m0":
            y_2d = gemm_mxfp8_nt(x_2d, w, block_k=block_k, out_dtype=x.dtype)
        elif mode == "fp8_hier_bf16_fp32":
            y_2d = gemm_hier_fp8_nt(x_2d, w, block_k=block_k, out_dtype=x.dtype)
        elif mode == "fp8_rowwise":
            y_2d = gemm_hipblaslt_fp8_nt(x_2d, w, out_dtype=x.dtype)
        else:
            y_2d = F.linear(x_2d, w)
        return y_2d.reshape(*orig_shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_2d, w = ctx.saved_tensors
        orig_shape, mode, block_k, x_dtype = ctx.meta
        dy_2d = dy.reshape(-1, w.shape[0]).contiguous()

        dx = dw = None
        if ctx.needs_input_grad[0]:
            wt = w.t().contiguous()
            if mode == "mxfp8_e8m0" and (w.shape[0] % block_k == 0):
                dx_2d = gemm_mxfp8_nt(dy_2d, wt, block_k=block_k, out_dtype=x_dtype)
            elif mode == "fp8_hier_bf16_fp32" and (w.shape[0] % block_k == 0):
                dx_2d = gemm_hier_fp8_nt(dy_2d, wt, block_k=block_k, out_dtype=x_dtype)
            elif mode == "fp8_rowwise":
                dx_2d = gemm_hipblaslt_fp8_nt(dy_2d, wt, out_dtype=x_dtype)
            else:
                dx_2d = torch.matmul(dy_2d, w)
            dx = dx_2d.reshape(orig_shape)

        if ctx.needs_input_grad[1]:
            M_k = dy_2d.shape[0]
            if mode == "mxfp8_e8m0" and (M_k % block_k == 0):
                dw = gemm_mxfp8_nt(dy_2d.t().contiguous(), x_2d.t().contiguous(), block_k=block_k, out_dtype=w.dtype)
            elif mode == "fp8_hier_bf16_fp32" and (M_k % block_k == 0):
                dw = gemm_hier_fp8_nt(dy_2d.t().contiguous(), x_2d.t().contiguous(), block_k=block_k, out_dtype=w.dtype)
            elif mode == "fp8_rowwise":
                dw = gemm_hipblaslt_fp8_nt(dy_2d.t().contiguous(), x_2d.t().contiguous(), out_dtype=w.dtype)
            else:
                dw = torch.matmul(dy_2d.t(), x_2d)

        return dx, dw, None, None


class QuantLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, mode: str = "bf16", block_k: int = 128, std: float = None):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.mode         = mode
        self.block_k      = block_k
        self.weight       = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.bfloat16))
        init_std = std if std is not None else (1.0 / math.sqrt(in_features))
        nn.init.normal_(self.weight, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "bf16":
            return F.linear(x, self.weight)
        return _QuantLinearFn.apply(x, self.weight, self.mode, self.block_k)
