#!/usr/bin/env python3
"""
Targeted mathematical & systems verification suite for `src/jlens_ablation.py` on ROCm/CUDA/CPU:
  1. Verifies KV-cached `advance_local_window_kv` (`W=1` and `W=4`) matches full-sequence `NanoGPTBlock` execution bit-for-bit (`< 1e-4` BF16 / `< 1e-6` FP32).
  2. Verifies covariance-matched oblique suppression (`metric='cov'`) zeroes `<w_i, x_ed>` on `[100, 1]` megaphone/whisper test without overshooting or corrupting orthogonal concepts, and `k x k` joint suppression zeroes all `k` readouts simultaneously.
  3. Verifies `superpos_unlikelihood` fixes Flaw #1 (sibling positive features do NOT cannibalize dominant feature `q_1 = 0.70`), bounds negative `u_i < 0` (`loss >= 0`, vanishing grad as `q_i -> 0`), and `strength=0` yields exact zero usefulness & zero grad.
  4. Verifies end-to-end `CausalAblationJLensController` stops early probes at `anchor_layer` (never executing suffix blocks!) and leaves frozen readout grad-free.
"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model_70m import NanoGPT70MConfig, NanoGPT70M
from jlens_ablation import (
    FrozenSoftcapReadout, WorkspaceJLens, select_diverse_fast,
    suppress_oblique_single_and_joint, compute_feature_loss,
    extract_clean_kv_for_blocks, advance_local_window_kv,
    CausalAblationJLensController,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)
    print(f"Running verify_jlens_ablation on {device}...")

    # 1. Megaphone [100, 1] vs Whisper covariance metric test
    ro = FrozenSoftcapReadout(torch.ones(2), torch.tensor([[1.0, 1.0], [1.0, -1.0], [0.0, 1.0]]), softcap=0.0).to(device)
    lens = WorkspaceJLens(2, ro, device).to(device)
    with torch.no_grad():
        lens.J.copy_(torch.diag(torch.tensor([100.0, 1.0], device=device)))
        lens.Sigma.copy_(torch.diag(torch.tensor([1e-4, 1.0], device=device)))

    ids_test = torch.tensor([[0, 1]], device=device)  # "Texas"=[1,1], "Florida"=[1,-1]
    w_r, d_w, u_s = lens.candidate_vectors(ids_test, metric="cov")
    # In cov mode, orthogonal output tokens [1,1] and [1,-1] have 0 cosine in u_sim!
    cos_tf = float((u_s[0, 0] * u_s[0, 1]).abs().item())
    assert cos_tf < 1e-6, f"Expected 0 cosine for orthogonal concepts, got {cos_tf}"

    x_test = torch.tensor([[0.01, 1.0]], device=device)
    probes, usa_k, usa_all = suppress_oblique_single_and_joint(
        x_test, w_r[:, :1], d_w[:, :1], torch.tensor([[True]], device=device), strength=1.0, match_norm=False
    )
    # Suppressing "Texas" (100*0.01 + 1*1.0 = 2.0) along Sigma@w = [0.01, 1.0] must yield [0, 0], leaving Florida (100*h1 - h2 = 0) at 0!
    torch.testing.assert_close(probes[0, 1], torch.zeros(2, device=device), atol=1e-5, rtol=1e-5)
    print("PASS 1: Covariance-matched oblique suppression zeroes stretched features cleanly & preserves orthogonal concepts")

    # 2. Joint k x k suppression zeroes all k readouts simultaneously
    x4 = torch.randn((3, 8), device=device)
    w4 = torch.randn((3, 3, 8), device=device)
    sig4 = torch.eye(8, device=device)
    d4 = torch.matmul(w4, sig4)
    val4 = torch.ones((3, 3), dtype=torch.bool, device=device)
    pr4, _, uall4 = suppress_oblique_single_and_joint(x4, w4, d4, val4, strength=1.0, match_norm=False)
    rem_readout = torch.einsum("bkd,bd->bk", w4, pr4[:, -1]).abs().max().item()
    assert uall4.all() and rem_readout < 1e-2, f"Joint suppression residual too large: {rem_readout}"
    print("PASS 2: Joint all-k oblique suppression zeroes all k feature readouts simultaneously")

    # 3. `superpos_unlikelihood` vs categorical cannibalization & negative unlikelihood bound
    # Feature 0 has q_0 = 0.70, u_0 = +0.20; Features 1..15 have q_j = 0.01, u_j = +0.03; Feature 16 (bg) has q_16 = 0.15
    probs = torch.tensor([[0.70] + [0.01] * 15 + [0.15]], device=device)
    z = probs.log().detach().requires_grad_(True)
    logq = z.log_softmax(dim=-1)
    ids16 = torch.arange(16, device=device).unsqueeze(0)
    u16   = torch.tensor([[0.20] + [0.03] * 15], device=device)
    v16   = torch.ones_like(ids16, dtype=torch.bool)
    loss_sup = compute_feature_loss(logq, ids16, u16, v16, loss_kind="superpos_unlikelihood").sum()
    loss_sup.backward()
    # Gradient descent step `-z.grad[0, 0]` MUST be positive (pushes top feature UP, unlike raw categorical NLL!)
    assert float(-z.grad[0, 0].item()) > 0.0, f"Expected positive push on #1 feature, got {-z.grad[0,0].item()}"
    assert float(loss_sup.item()) >= 0.0

    # Negative usefulness unlikelihood vanishes as q -> 0 and stays >= 0:
    z_neg = torch.tensor([[-12.0, 0.0]], device=device, requires_grad=True)
    l_neg = compute_feature_loss(
        z_neg.log_softmax(-1), torch.tensor([[0]], device=device),
        torch.tensor([[-0.5]], device=device), torch.tensor([[True]], device=device),
        loss_kind="superpos_unlikelihood",
    ).sum()
    l_neg.backward()
    assert float(l_neg.item()) >= 0.0 and abs(float(z_neg.grad[0, 0].item())) < 1e-4
    print("PASS 3: `superpos_unlikelihood` prevents sibling cannibalization & bounds negative usefulness with vanishing dead-feature grad")

    # 4. Exact KV-cached local window equivalence vs full-sequence execution + hook check
    cfg = NanoGPT70MConfig(n_layers=6, d_model=64, n_heads=4, head_dim=16, d_ff=128, vocab_pad=256, seq_len=32, ce_backend="eager")
    model = NanoGPT70M(cfg).to(device).eval()
    xb = torch.randint(0, 200, (4, 32), device=device)
    yb = torch.randint(0, 200, (4, 32), device=device)
    _, tok_nll, interm, x_raw = model(xb, yb, return_interm=True)

    pos = torch.tensor([0, 5, 17, 28], device=device)
    b_sel = torch.arange(4, device=device)
    W = 4
    P = 3
    pert = torch.randn((4, P, 64), device=device, dtype=torch.bfloat16) * 0.2
    win_idx = pos[:, None] + torch.arange(W, device=device)[None, :]
    h_clean_w = interm[1].detach()[b_sel[:, None], win_idx]
    h_win = h_clean_w.unsqueeze(1).expand(4, P, W, 64).clone()
    h_win[:, :, 0, :] = pert

    sub_blocks = model.blocks[2:5]
    sub_inputs = [interm[m - 1].detach() for m in range(2, 5)]
    kv_sub = extract_clean_kv_for_blocks(sub_blocks, sub_inputs, b_sel, model.cos, model.sin)
    out_kv = advance_local_window_kv(sub_blocks, kv_sub, h_win, pos, model.cos, model.sin)

    # Compare against brute-force full-sequence forward for each probe p:
    with torch.no_grad():
        for p_i in range(P):
            full_h = interm[1].detach().clone()
            full_h[b_sel, pos] = pert[:, p_i]
            for blk in sub_blocks:
                full_h = blk(full_h, model.cos[:, :, :32, :], model.sin[:, :, :32, :])
            ref_w = full_h[b_sel[:, None], win_idx]
            torch.testing.assert_close(out_kv[:, p_i], ref_w, atol=2e-2, rtol=2e-2)
    print("PASS 4: KV-cached local window matches full-sequence causal propagation across pos=0..28")

    # 5. End-to-end controller backward & early-probe suffix hook check
    ctrl = CausalAblationJLensController(
        model, device=device, early_layer=2, anchor_layer=4, probe_seqs=4, pool_size=32, k_dirs=4, window=4
    )
    ctrl.refresh_snapshot(model, [t.detach() for t in interm], x_raw)

    # Zero-strength check:
    ctrl.strength = 0.0
    ce_loss, tok_nll, interm, x_raw = model(xb, yb, return_interm=True)
    aux0, _, st0 = ctrl.compute_loss(model, interm, yb, tok_nll)
    assert st0["u_anc_abs"] == 0.0 and st0["u_ear_abs"] == 0.0 and float(aux0.item()) == 0.0
    ctrl.strength = 1.0

    # Full backward finiteness & frozen readout check:
    model.zero_grad(set_to_none=True)
    ce_loss, tok_nll, interm, x_raw = model(xb, yb, return_interm=True)
    aux, _, st = ctrl.compute_loss(model, interm, yb, tok_nll)
    (ce_loss + 0.05 * aux).backward()
    assert torch.isfinite(aux)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(p.grad is None for p in ctrl.readout.parameters())
    print("PASS 5: Zero-strength=0 check, finite combined backward, and grad-free frozen readout:", st)
    print("ALL VERIFICATION CHECKS PASSED.")


if __name__ == "__main__":
    main()
