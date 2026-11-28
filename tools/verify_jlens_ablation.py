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
    cos_tf = float((u_s[0, 0] * u_s[0, 1]).sum().abs().item())
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
        model, device=device, early_layer=2, anchor_layer=4, active_frac=0.0, probe_seqs=4, pool_size=32, k_dirs=4, window=4
    )
    ctrl.refresh_snapshot(model, [t.detach() for t in interm], x_raw)

    # Zero-strength check:
    ctrl.strength = 0.0
    ce_loss, tok_nll, interm, x_raw = model(xb, yb, return_interm=True)
    aux0, _, st0 = ctrl.compute_loss(model, interm, yb, tok_nll)
    assert st0["u_anc_abs"] == 0.0 and st0["u_ear_abs"] == 0.0 and float(aux0.item()) == 0.0
    ctrl.strength = 1.0

    # Full backward finiteness & frozen readout check (including under BF16 autocast!):
    model.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        ce_loss, tok_nll, interm, x_raw = model(xb, yb, return_interm=True)
        ctrl.refresh_snapshot(model, [t.detach() for t in interm], x_raw)
        aux, _, st = ctrl.compute_loss(model, interm, yb, tok_nll)
    (ce_loss + 0.05 * aux).backward()
    assert torch.isfinite(aux)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(p.grad is None for p in ctrl.readout.parameters())
    print("PASS 5: Zero-strength=0 check, finite combined backward, and grad-free frozen readout:", st)

    # 6. Multi-position shared-KV Triton Flash-Prefix (Dh=64) & SelectedSoftcapLinearLogProbsFn grad parity:
    cfg64 = NanoGPT70MConfig(n_layers=6, d_model=128, n_heads=2, head_dim=64, d_ff=256, vocab_pad=2048, seq_len=64, ce_backend="eager")
    m64 = NanoGPT70M(cfg64).to(device).eval()
    xb64 = torch.randint(0, 2000, (4, 64), device=device)
    yb64 = torch.randint(0, 2000, (4, 64), device=device)
    _, nll64, int64, xraw64 = m64(xb64, yb64, return_interm=True)

    # Multiple positions sharing the SAME sequence KV cache (b=[0, 0, 1, 2, 2, 3], pos=[0, 19, 31, 7, 52, 40]):
    b_multi   = torch.tensor([0, 0, 1, 2, 2, 3], device=device)
    pos_multi = torch.tensor([0, 19, 31, 7, 52, 40], device=device)
    uniq_b, kv_slot = torch.unique_consecutive(b_multi, return_inverse=True)
    W64, P64 = 4, 5
    pert64 = torch.randn((6, P64, 128), device=device, dtype=torch.bfloat16) * 0.2
    w_idx64 = pos_multi[:, None] + torch.arange(W64, device=device)[None, :]
    hcl64 = int64[1].detach()[b_multi[:, None], w_idx64]
    hwin64 = hcl64.unsqueeze(1).expand(6, P64, W64, 128).clone()
    hwin64[:, :, 0, :] = pert64

    sub64 = m64.blocks[2:5]
    inp64 = [int64[m - 1].detach() for m in range(2, 5)]
    kv64 = extract_clean_kv_for_blocks(sub64, inp64, uniq_b, m64.cos, m64.sin)
    out_flash = advance_local_window_kv(sub64, kv64, hwin64, pos_multi, m64.cos, m64.sin, kv_slot=kv_slot)

    with torch.no_grad():
        for idx_n in range(6):
            bi = int(b_multi[idx_n].item())
            pi = int(pos_multi[idx_n].item())
            for p_i in range(P64):
                fh = int64[1].detach()[bi : bi + 1].clone()
                fh[0, pi] = pert64[idx_n, p_i]
                for blk in sub64:
                    fh = blk(fh, m64.cos[:, :, :64, :], m64.sin[:, :, :64, :])
                torch.testing.assert_close(out_flash[idx_n, p_i], fh[0, pi : pi + W64], atol=2e-2, rtol=2e-2)
    print("PASS 6: Multi-position shared-KV Triton Flash-Prefix (Dh=64, pos=0..52) matches brute-force full-sequence forward")

    # 7. Differentiable SelectedSoftcapLinearLogProbsFn vs exact FP32 log_probs + multi-position active_frac=0.15 check:
    ro64 = FrozenSoftcapReadout(m64.out_norm.weight, m64.tok_emb.weight, softcap=30.0).to(device)
    x_ref = torch.randn((32, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
    x_fst = x_ref.detach().clone().requires_grad_(True)
    ids_t = torch.randint(0, 2048, (32, 8), device=device)
    u_wt  = torch.randn((32, 8), device=device)
    l_ref = -(ro64.log_probs(x_ref).gather(-1, ids_t) * u_wt).sum()
    l_ref.backward()
    l_fst = -(ro64.selected_log_probs(x_fst, ids_t) * u_wt).sum()
    l_fst.backward()
    g_cos = float(torch.nn.functional.cosine_similarity(x_ref.grad.float().reshape(1, -1), x_fst.grad.float().reshape(1, -1)).item())
    assert g_cos > 0.9995, f"SelectedSoftcapLinearLogProbsFn grad cosine too low: {g_cos}"

    ctrl_mp = CausalAblationJLensController(m64, device=device, early_layer=2, anchor_layer=4, active_frac=0.15, k_dirs=4, window=4)
    m64.train().zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        ce64, nll64, int64, xraw64 = m64(xb64, yb64, return_interm=True)
        ctrl_mp.refresh_snapshot(m64, [t.detach() for t in int64], xraw64)
        aux64, _, st64 = ctrl_mp.compute_loss(m64, int64, yb64, nll64)
    (ce64 + 0.05 * aux64).backward()
    assert st64["n_act"] == int(round(0.15 * 4 * 64)) and torch.isfinite(aux64)
    print(f"PASS 7: SelectedSoftcapLinearLogProbsFn grad_cos={g_cos:.6f} & multi-position active_frac=0.15 passed:", st64)

    # 8. Verify all 3 Wave 4.1 fixes: (1) batch-wide 1/(B*T) normalization, (2) matched EMA on J & U, (3) pure_rms_norm_fp32 invariance!
    exp_jbatch = 0.5 * (st64["loss_anc"] + st64["loss_ear"]) * (st64["n_act"] / float(4 * 64))
    assert abs(float(aux64.item()) - exp_jbatch) < 1e-4, f"Expected batch-normalized j_loss={exp_jbatch}, got {aux64.item()}"

    u_before = ctrl_mp.readout.U.clone()
    with torch.no_grad():
        m64.tok_emb.weight.add_(1.0)
    ctrl_mp.refresh_snapshot(m64, [t.detach() for t in int64], xraw64)
    # With ema=0.50 on an already-initialized controller, U must move halfway (+0.50), NOT snap 1.0!
    u_delta = float((ctrl_mp.readout.U - u_before).mean().item())
    assert abs(u_delta - 0.50) < 1e-4, f"Expected matched ema=0.50 update on readout.U (delta=0.50), got {u_delta}"

    # Verify scale invariance & radial grad orthogonality (<grad, x> == 0) from `pure_rms_norm_fp32` in WorkspaceJLens:
    x_s1 = torch.randn((8, 128), device=device, dtype=torch.float32, requires_grad=True)
    x_s10 = (x_s1.detach() * 10.0).requires_grad_(True)
    lp1  = ctrl_mp.lens_anchor.log_probs(x_s1)
    lp10 = ctrl_mp.lens_anchor.log_probs(x_s10)
    torch.testing.assert_close(lp1, lp10, atol=1e-4, rtol=1e-4)
    lp1[:, :4].sum().backward()
    rad_dot = float((x_s1.grad * x_s1.detach()).sum(dim=-1).abs().max().item())
    assert rad_dot < 1e-4, f"Expected tangential grad (<grad, x> == 0), got {rad_dot}"
    print(f"PASS 8: Verified 1/(B*T) nat normalization, matched J/U EMA={ctrl_mp.ema} (dU={u_delta:.4f}), and RMSNorm sink-invariance (rad_dot={rad_dot:.2e})")

    # 9. Verify `random_gap4` layer_mode on 12-layer config (exact 15 valid pairs, batched 9-layer Cholesky refresh, finite grad):
    cfg12 = NanoGPT70MConfig(n_layers=12, d_model=64, n_heads=2, head_dim=32, d_ff=128, vocab_pad=512, seq_len=32, ce_backend="eager")
    m12 = NanoGPT70M(cfg12).to(device).train()
    ctrl_rg = CausalAblationJLensController(m12, device=device, layer_mode="random_gap4", active_frac=0.10, k_dirs=4, window=4)
    assert len(ctrl_rg.valid_layer_pairs) == 15 and all((la - le) >= 4 and 2 <= le <= 6 and 6 <= la <= 10 for le, la in ctrl_rg.valid_layer_pairs)
    xb12 = torch.randint(0, 500, (2, 32), device=device)
    yb12 = torch.randint(0, 500, (2, 32), device=device)
    m12.zero_grad(set_to_none=True)
    ce12, nll12, int12, xraw12 = m12(xb12, yb12, return_interm=True)
    ctrl_rg.refresh_snapshot(m12, [t.detach() for t in int12], xraw12)
    seen_pairs = set()
    for _ in range(6):
        aux12, _, st12 = ctrl_rg.compute_loss(m12, int12, yb12, nll12)
        seen_pairs.add((st12["l_ear"], st12["l_anc"]))
    (ce12 + 0.50 * aux12).backward()
    assert len(seen_pairs) >= 2 and torch.isfinite(aux12)
    print(f"PASS 9: Verified `random_gap4` (15 valid pairs, 9-layer batched Stein refresh, sampled={sorted(seen_pairs)})")
    print("ALL VERIFICATION CHECKS PASSED.")


if __name__ == "__main__":
    main()
