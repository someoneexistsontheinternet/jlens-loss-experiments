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
    jlens_mode: str     = "none"   # "none" | "wave8"
    jlens_weight: float = 0.35


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
            dw.addmm_(dl.t(), hc, beta=1.0, alpha=1.0)

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


class NanoGPTBlock(nn.Module):
    def __init__(self, cfg: NanoGPT70MConfig):
        super().__init__()
        self.cfg = cfg
        D, F_d = cfg.d_model, cfg.d_ff
        res_std  = (1.0 / math.sqrt(2.0 * cfg.n_layers)) / math.sqrt(D)
        down_std = (1.0 / math.sqrt(2.0 * cfg.n_layers)) / math.sqrt(F_d)

        self.attn_norm = nn.RMSNorm(D)
        self.wqkv      = QuantLinear(D, 3 * D, mode=cfg.linear_mode, block_k=cfg.fp8_block_k)
        self.q_norm    = nn.RMSNorm(cfg.head_dim)
        self.k_norm    = nn.RMSNorm(cfg.head_dim)
        self.wo        = QuantLinear(D, D, mode=cfg.linear_mode, block_k=cfg.fp8_block_k, std=res_std)

        self.mlp_norm  = nn.RMSNorm(D)
        self.wgu       = QuantLinear(D, 2 * F_d, mode=cfg.linear_mode, block_k=cfg.fp8_block_k)
        self.wdown     = QuantLinear(F_d, D, mode=cfg.linear_mode, block_k=cfg.fp8_block_k, std=down_std)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh   = self.cfg.n_heads, self.cfg.head_dim
        qkv = self.wqkv(self.attn_norm(x)).view(B, T, 3, H, Dh)
        q   = apply_rope(self.q_norm(qkv[:, :, 0]).transpose(1, 2), cos, sin)
        k   = apply_rope(self.k_norm(qkv[:, :, 1]).transpose(1, 2), cos, sin)
        v   = qkv[:, :, 2].transpose(1, 2)
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
        self.out_norm = nn.RMSNorm(cfg.d_model)
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
        loss, tok_nll = ChunkedSoftcapCEFn.apply(
            h.reshape(-1, self.cfg.d_model), self.tok_emb.weight, targets.reshape(-1), float(self.cfg.softcap), 8192
        )
        if return_interm:
            return loss, tok_nll.detach(), interm, x_raw.detach()
        return loss, tok_nll.detach()
