"""
71.38M-Parameter 256-Aligned NanoGPT Model for MI325X:
  - 12 Layers, D = 512 (2x256), 8 Heads x 64, SwiGLU d_ff = 1792 (7x256), Vocab = 50,304 (393x128).
  - Supports pluggable `linear_mode` ("bf16", "fp8_rowwise", "mxfp8_e8m0", "fp8_hier_bf16_fp32").
  - Zero-HBM-spill chunked logit-softcapped Cross-Entropy returning detached per-token NLL.
  - Supports 8-Horizon Focusing-Wave Stein `J-Lens` (`return_interm=True`).
"""
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from fp8_kernels import QuantLinear


@dataclass
class NanoGPT70MConfig:
    n_layers: int       = 12
    d_model: int        = 512      # Divisible by 64, 128, 256
    n_heads: int        = 8
    head_dim: int       = 64
    d_ff: int           = 1792     # 7 x 256 -> 71.38M total params with tied 50,304 vocab!
    vocab_pad: int      = 50304    # 393 x 128
    seq_len: int        = 1024
    softcap: float      = 30.0
    linear_mode: str    = "bf16"   # "bf16" | "fp8_rowwise" | "mxfp8_e8m0" | "fp8_hier_bf16_fp32"
    fp8_block_k: int    = 128
    e8m0_mode: str      = "ceil_240"
    hier_w_2d: bool     = False
    jlens_mode: str     = "none"   # "none" | "wave8" | "ablation"
    jlens_weight: float = 0.35
    ce_backend: str     = "triton" # "triton" (fused in-place row softcap CE) | "eager"
    ce_chunk_sz: int    = 4096     # Infinity-Cache resident (< 256 MB MI325X LLC!)
    fused_block_ops: bool = True   # Fused Triton QKNorm+RoPE+BHTD & SwiGLU


@triton.jit
def _softcap_ce_fwd_bwd_inplace_kernel(
    logits_ptr, stride_m, targets_ptr, nll_ptr,
    V: tl.constexpr, SOFTCAP: tl.constexpr, INV_SOFTCAP: tl.constexpr,
    INV_M, BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * stride_m
    tgt = tl.load(targets_ptr + row)

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
            th = (e2z - 1.0) / (e2z + 1.0)
            lg = tl.where(mask_v, SOFTCAP * th, -float("inf"))
        else:
            lg = raw
        blk_max = tl.max(lg, axis=0)
        m_new = tl.maximum(m_i, blk_max)
        d_i = d_i * tl.math.exp2((m_i - m_new) * LOG2E) + tl.sum(tl.math.exp2((lg - m_new) * LOG2E), axis=0)
        m_i = m_new

    lse = m_i + tl.math.log(d_i)
    raw_t = tl.load(row_ptr + tgt).to(tl.float32)
    if SOFTCAP > 0.0:
        zt = tl.minimum(tl.maximum(raw_t * INV_SOFTCAP, -20.0), 20.0)
        e2zt = tl.math.exp2(2.0 * LOG2E * zt)
        lg_t = SOFTCAP * ((e2zt - 1.0) / (e2zt + 1.0))
    else:
        lg_t = raw_t
    tl.store(nll_ptr + row, lse - lg_t)

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
        grad = (prob - tl.where(offs_v == tgt, 1.0, 0.0)) * (ds * INV_M)
        tl.store(row_ptr + offs_v, grad.to(tl.bfloat16), mask=mask_v)


class FusedTritonChunkedSoftcapCEFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h_2d: torch.Tensor, w_vocab: torch.Tensor, targets_1d: torch.Tensor, softcap: float, chunk_sz: int = 32768):
        M, D = h_2d.shape
        V = w_vocab.shape[0]
        inv_m = 1.0 / float(M)

        dh = torch.empty((M, D), dtype=torch.bfloat16, device=h_2d.device)
        dw = torch.zeros_like(w_vocab, dtype=torch.float32)
        nll_out = torch.empty((M,), dtype=torch.float32, device=h_2d.device)

        w_bf = w_vocab.to(torch.bfloat16)
        inv_sc = (1.0 / softcap) if softcap > 0.0 else 0.0
        max_rows = min(M, chunk_sz)
        logits_buf = torch.empty((max_rows, V), dtype=torch.bfloat16, device=h_2d.device)

        for s in range(0, M, chunk_sz):
            e = min(s + chunk_sz, M)
            n_r = e - s
            hc = h_2d[s:e].to(torch.bfloat16)
            tc = targets_1d[s:e]
            buf = logits_buf[:n_r]
            torch.matmul(hc, w_bf.t(), out=buf)
            _softcap_ce_fwd_bwd_inplace_kernel[(n_r,)](
                buf, buf.stride(0), tc, nll_out[s:e],
                V=V, SOFTCAP=float(softcap), INV_SOFTCAP=float(inv_sc),
                INV_M=inv_m, BLOCK_V=4096, num_warps=8,
            )
            torch.matmul(buf, w_bf, out=dh[s:e])
            dw.add_(torch.matmul(buf.t(), hc).float())

        loss_acc = nll_out.mean()
        ctx.h_dtype = h_2d.dtype
        ctx.save_for_backward(dh, dw.to(w_vocab.dtype))
        return loss_acc, nll_out

    @staticmethod
    def backward(ctx, grad_loss, _grad_nll):
        dh, dw = ctx.saved_tensors
        g = grad_loss.to(ctx.h_dtype)
        return dh.to(ctx.h_dtype) * g, dw * grad_loss.to(dw.dtype), None, None, None


class FastRMSNorm(nn.Module):
    """FP32-accumulated RMSNorm that preserves input dtype (BF16) for FlashAttention & QuantLinear."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight).to(x.dtype)


class ChunkedSoftcapCEFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h_2d: torch.Tensor, w_vocab: torch.Tensor, targets_1d: torch.Tensor, softcap: float, chunk_sz: int = 8192):
        M, D = h_2d.shape
        inv_m = 1.0 / float(M)

        dh = torch.empty_like(h_2d)
        dw = torch.zeros_like(w_vocab, dtype=torch.float32)
        nll_out = torch.empty((M,), dtype=torch.float32, device=h_2d.device)
        loss_acc = torch.zeros((), dtype=torch.float32, device=h_2d.device)

        w_bf = w_vocab.to(torch.bfloat16)
        for s in range(0, M, chunk_sz):
            e = min(s + chunk_sz, M)
            hc = h_2d[s:e].to(torch.bfloat16)
            tc = targets_1d[s:e]
            raw = torch.matmul(hc, w_bf.t()).float()
            if softcap > 0.0:
                th = torch.tanh(raw * (1.0 / softcap))
                lg = softcap * th
                ds = 1.0 - th * th
            else:
                lg = raw
                ds = 1.0
            lse = torch.logsumexp(lg, dim=-1)
            tgt = lg.gather(1, tc.unsqueeze(1)).squeeze(1)
            nll = lse - tgt
            nll_out[s:e] = nll
            loss_acc += nll.sum() * inv_m

            probs = torch.exp(lg - lse.unsqueeze(1))
            probs.scatter_add_(1, tc.unsqueeze(1), torch.full((e - s, 1), -1.0, device=h_2d.device, dtype=torch.float32))
            dl = (probs * (ds * inv_m)).to(torch.bfloat16)
            dh[s:e] = torch.matmul(dl, w_bf).to(h_2d.dtype)
            dw.add_(torch.matmul(dl.t(), hc).float())

        ctx.save_for_backward(dh, dw.to(w_vocab.dtype))
        return loss_acc, nll_out

    @staticmethod
    def backward(ctx, grad_loss, _grad_nll):
        dh, dw = ctx.saved_tensors
        g = grad_loss.to(dh.dtype)
        return dh * g, dw * grad_loss.to(dw.dtype), None, None, None


def apply_rope(x_bhtd: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x_bhtd.shape[-1] // 2
    x1, x2 = x_bhtd[..., :half], x_bhtd[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


@torch._dynamo.disable
def _dispatch_chunked_softcap_ce(
    h_2d: torch.Tensor, w_vocab: torch.Tensor, targets_1d: torch.Tensor,
    softcap: float, backend: str, chunk_sz: int,
):
    if backend == "triton":
        from fast_kernels import FastChunkedSoftcapCEFn
        return FastChunkedSoftcapCEFn.apply(h_2d, w_vocab, targets_1d, softcap, chunk_sz, True)
    return ChunkedSoftcapCEFn.apply(h_2d, w_vocab, targets_1d, softcap, chunk_sz)


class NanoGPTBlock(nn.Module):
    def __init__(self, cfg: NanoGPT70MConfig):
        super().__init__()
        self.cfg = cfg
        D, F_d = cfg.d_model, cfg.d_ff
        res_std  = (1.0 / math.sqrt(2.0 * cfg.n_layers)) / math.sqrt(D)
        down_std = (1.0 / math.sqrt(2.0 * cfg.n_layers)) / math.sqrt(F_d)

        kw = dict(mode=cfg.linear_mode, block_k=cfg.fp8_block_k, e8m0_mode=cfg.e8m0_mode, hier_w_2d=cfg.hier_w_2d)
        self.attn_norm = FastRMSNorm(D)
        self.wqkv      = QuantLinear(D, 3 * D, **kw)
        self.q_norm    = FastRMSNorm(cfg.head_dim)
        self.k_norm    = FastRMSNorm(cfg.head_dim)
        self.wo        = QuantLinear(D, D, std=res_std, **kw)

        self.mlp_norm  = FastRMSNorm(D)
        self.wgu       = QuantLinear(D, 2 * F_d, **kw)
        self.wdown     = QuantLinear(F_d, D, std=down_std, **kw)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh   = self.cfg.n_heads, self.cfg.head_dim
        qkv = self.wqkv(self.attn_norm(x)).view(B, T, 3, H, Dh)
        if self.cfg.fused_block_ops and x.is_cuda and Dh == 64:
            from fast_kernels import FusedQKNormRoPEFn, FusedSwiGLUFn
            q, k, v = FusedQKNormRoPEFn.apply(qkv, self.q_norm.weight, self.k_norm.weight, cos, sin, self.q_norm.eps)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)
            x = x + self.wo(y)
            x = x + self.wdown(FusedSwiGLUFn.apply(self.wgu(self.mlp_norm(x))))
            return x
        q   = apply_rope(self.q_norm(qkv[:, :, 0]).transpose(1, 2), cos, sin)
        k   = apply_rope(self.k_norm(qkv[:, :, 1]).transpose(1, 2), cos, sin)
        v   = qkv[:, :, 2].transpose(1, 2).contiguous()
        y   = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)
        x   = x + self.wo(y)

        g, u = self.wgu(self.mlp_norm(x)).chunk(2, dim=-1)
        x    = x + self.wdown(F.silu(g) * u)
        return x


class NanoGPT70M(nn.Module):
    def __init__(self, cfg: NanoGPT70MConfig):
        super().__init__()
        self.cfg      = cfg
        self.tok_emb  = nn.Embedding(cfg.vocab_pad, cfg.d_model)
        self.blocks   = nn.ModuleList([NanoGPTBlock(cfg) for _ in range(cfg.n_layers)])
        self.out_norm = FastRMSNorm(cfg.d_model)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)

        half = cfg.head_dim // 2
        inv_f = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
        freqs = torch.outer(torch.arange(cfg.seq_len, dtype=torch.float32), inv_f)
        self.register_buffer("cos", torch.cos(freqs)[None, None, :, :].to(torch.bfloat16), persistent=False)
        self.register_buffer("sin", torch.sin(freqs)[None, None, :, :].to(torch.bfloat16), persistent=False)

    def count_params_and_flops(self):
        n_tot = sum(p.numel() for p in self.parameters())
        n_emb = self.tok_emb.weight.numel()
        n_lin = sum(m.weight.numel() for m in self.modules() if isinstance(m, QuantLinear))
        flops_tok = 6 * n_lin + 6 * n_emb + self.cfg.n_layers * 3 * (4 * self.cfg.d_model * self.cfg.seq_len)
        return n_tot, n_tot - n_emb, flops_tok

    def forward_backbone(self, idx: torch.Tensor, return_interm: bool = False):
        T = idx.shape[1]
        cos, sin = self.cos[:, :, :T, :], self.sin[:, :, :T, :]
        x = self.tok_emb(idx).to(torch.bfloat16)
        interm = []
        for l_i, blk in enumerate(self.blocks):
            x = blk(x, cos, sin)
            if return_interm and l_i < self.cfg.n_layers - 1:
                interm.append(x)
        h_fin = self.out_norm(x)
        return (h_fin, interm, x) if return_interm else h_fin

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, return_interm: bool = False):
        if return_interm:
            h, interm, x_raw = self.forward_backbone(idx, return_interm=True)
        else:
            h = self.forward_backbone(idx, return_interm=False)
        if targets is None:
            logits = F.linear(h, self.tok_emb.weight.to(h.dtype))
            if self.cfg.softcap > 0.0:
                logits = self.cfg.softcap * torch.tanh(logits.float() * (1.0 / self.cfg.softcap))
            return logits
        loss, tok_nll = _dispatch_chunked_softcap_ce(
            h.reshape(-1, self.cfg.d_model), self.tok_emb.weight, targets.reshape(-1),
            float(self.cfg.softcap), self.cfg.ce_backend, int(self.cfg.ce_chunk_sz),
        )
        if return_interm:
            return loss, tok_nll.detach(), interm, x_raw.detach()
        return loss, tok_nll.detach()
