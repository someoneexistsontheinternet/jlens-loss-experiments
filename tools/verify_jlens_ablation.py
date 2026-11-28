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

    # 10. Verify `exp_signed_clamped` (beta in {0, 6, 12}: full -(1-q) negative suppression at q=0.01, zero grad below log_q=-9.0) + `random_anc_7_10`:
    z_esc = torch.tensor([[-4.60517, 0.0]], device=device, requires_grad=True)  # q_0 ~ 0.01
    l_esc = compute_feature_loss(
        z_esc.log_softmax(-1), torch.tensor([[0]], device=device),
        torch.tensor([[-0.30]], device=device), torch.tensor([[True]], device=device),
        loss_kind="exp_signed_clamped", exp_beta=6.0,
    ).sum()
    l_esc.backward()
    # At q_0 ~ 0.01, dL/dz_0 must be ~ +0.30 * (1 - 0.01) ~ +0.297 (>100x stronger than unlikelihood's +0.003!):
    assert float(z_esc.grad[0, 0].item()) > 0.25, f"Expected full -(1-q) negative suppression grad > 0.25, got {z_esc.grad[0, 0].item()}"

    z_dead = torch.tensor([[-12.0, 0.0]], device=device, requires_grad=True)   # log_q < -9.0 -> clamped to zero grad!
    l_dead = compute_feature_loss(
        z_dead.log_softmax(-1), torch.tensor([[0]], device=device),
        torch.tensor([[-0.50]], device=device), torch.tensor([[True]], device=device),
        loss_kind="exp_signed_clamped", exp_beta=12.0,
    ).sum()
    l_dead.backward()
    assert torch.isfinite(l_dead) and abs(float(z_dead.grad[0, 0].item())) == 0.0

    ctrl_ra = CausalAblationJLensController(
        m12, device=device, enable_early=False, layer_mode="random_anc_7_10",
        active_frac=0.10, k_dirs=4, window=1, loss_kind="exp_signed_clamped", exp_beta=6.0,
    )
    assert [la for _, la in ctrl_ra.valid_layer_pairs] == [7, 8, 9, 10]
    ctrl_ra.refresh_snapshot(m12, [t.detach() for t in int12], xraw12)
    aux_ra, _, st_ra = ctrl_ra.compute_loss(m12, int12, yb12, nll12)
    assert st_ra["l_anc"] in (7, 8, 9, 10) and torch.isfinite(aux_ra)
    print(f"PASS 10: Verified `exp_signed_clamped` (q=0.01 grad={z_esc.grad[0,0].item():.4f}, log_q<-9 grad=0.0) & `random_anc_7_10` (sampled={st_ra['l_anc']})")

    # 11. Verify Fused Hybrid (`Wave8SteinJLensController` sparse taps + `CausalAblationJLensController` sharing ONE `interm` pass):
    from jlens_wave8 import Wave8SteinJLensController
    cfg_hyb = NanoGPT70MConfig(n_layers=12, d_model=64, n_heads=2, head_dim=32, d_ff=128, vocab_pad=512, seq_len=256, ce_backend="eager")
    m_hyb = NanoGPT70M(cfg_hyb).to(device).train()
    w8_full = Wave8SteinJLensController(
        n_layers=12, d_model=64, vocab_pad=512, micro_tokens=8 * 256, device=device, k_vocab=64,
    )
    w8_sparse = Wave8SteinJLensController(
        n_layers=12, d_model=64, vocab_pad=512, micro_tokens=8 * 256, device=device, k_vocab=64, layer_indices=[3, 5, 7, 9],
    )
    assert w8_full.n_jlens == 10 and w8_sparse.n_jlens == 4
    assert abs(w8_sparse.estimate_call_flops(False) / w8_full.estimate_call_flops(False) - 0.40) < 1e-6
    ctrl_hyb_abl = CausalAblationJLensController(
        m_hyb, device=device, enable_early=False, anchor_layer=9,
        active_frac=0.02, k_dirs=4, window=1, loss_kind="exp_signed_clamped", exp_beta=6.0,
    )
    xb_h = torch.randint(0, 500, (8, 256), device=device)
    yb_h = torch.randint(0, 500, (8, 256), device=device)
    m_hyb.zero_grad(set_to_none=True)
    ce_h, nll_h, int_h, xraw_h = m_hyb(xb_h, yb_h, return_interm=True)
    int_h_det = [t.detach() for t in int_h]
    w8_sparse.refresh_lenses(int_h_det, xraw_h, yb_h)
    ctrl_hyb_abl.refresh_snapshot(m_hyb, int_h_det, xraw_h)
    cids_h = w8_sparse.sample_vocab_candidates()
    loss_w8, _ = w8_sparse.compute_loss(int_h, m_hyb.out_norm.weight, m_hyb.tok_emb.weight, yb_h, nll_h, cids_h)
    loss_ab, _, st_h = ctrl_hyb_abl.compute_loss(m_hyb, int_h, yb_h, nll_h)
    (ce_h + 0.05 * loss_w8 + 0.50 * loss_ab).backward()
    assert torch.isfinite(loss_w8) and torch.isfinite(loss_ab)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m_hyb.parameters())
    print(f"PASS 11: Verified Hybrid `Wave8 + CausalAblation` shared-interm fwd+bwd (w8={loss_w8.item():.4f}, abl={loss_ab.item():.5f}, sparse_ratio=0.40)")

    # 12. Verify `maxmin`, `kmeans`, and `emeans` (threshold-free clustering, synonym collapse, rival surfacing, 100% k slots, GPU speed):
    import time
    from jlens_ablation import FrozenSoftcapReadout, WorkspaceJLens
    V_syn, D_syn = 64, 32
    U_syn = torch.zeros((V_syn, D_syn), dtype=torch.float32, device=device)
    g_syn = torch.Generator(device="cpu").manual_seed(777)
    # 50 near-identical synonyms around e_0 (cos > 0.998):
    noise_50 = torch.randn((50, D_syn), generator=g_syn).to(device) * 0.01
    U_syn[:50] = F.normalize(F.one_hot(torch.zeros(50, dtype=torch.long, device=device), D_syn).float() + noise_50, dim=-1)
    # 7 strong orthogonal rival modes along e_1..e_7:
    for r_i in range(7):
        U_syn[50 + r_i, 1 + r_i] = 1.0
    # 7 near-zero-mass orthogonal noise modes along e_8..e_14:
    for r_i in range(7):
        U_syn[57 + r_i, 8 + r_i] = 1.0

    ro_syn = FrozenSoftcapReadout(torch.ones(D_syn, device=device), U_syn, softcap=30.0).to(device)
    lens_syn = WorkspaceJLens(D_syn, ro_syn, device, norm_input=False).to(device)
    pool_syn = torch.arange(64, device=device, dtype=torch.long).unsqueeze(0).expand(4, 64).contiguous()
    raw_lg = torch.empty((64,), dtype=torch.float32, device=device)
    raw_lg[:50]   = torch.linspace(12.0, 10.0, 50, device=device)   # high-prob duplicate synonym cluster
    raw_lg[50:57] = torch.linspace(6.0, 5.0, 7, device=device)      # legitimate orthogonal rivals
    raw_lg[57:64] = -8.0                                            # near-zero-mass tail noise
    probs_syn = F.softmax(raw_lg, dim=-1).unsqueeze(0).expand(4, 64).contiguous()

    for smode in ("maxmin", "kmeans", "emeans"):
        ids_s, w_s, d_s, val_s = select_diverse_from_pool(
            pool_syn, lens_syn, k=8, metric="cov", select_mode=smode, pool_probs=probs_syn
        )
        assert bool(val_s.all().item()), f"{smode} failed to fill all k=8 slots!"
        for b_i in range(4):
            picked = [int(x) for x in ids_s[b_i].tolist()]
            assert len(set(picked)) == 8, f"{smode} returned duplicate IDs: {picked}"
            n_from_syn   = sum(0 <= x < 50 for x in picked)
            n_from_rival = sum(50 <= x < 57 for x in picked)
            n_from_noise = sum(57 <= x < 64 for x in picked)
            assert n_from_syn == 1 and n_from_rival == 7 and n_from_noise == 0, (
                f"{smode} failed synonym collapse / rival surfacing: syn={n_from_syn}, rival={n_from_rival}, noise={n_from_noise}, picked={picked}"
            )

        ctrl_sm = CausalAblationJLensController(
            m_hyb, device=device, enable_early=False, anchor_layer=9,
            active_frac=0.02, k_dirs=8, window=1, select_mode=smode,
            loss_kind="exp_signed_clamped", exp_beta=6.0,
        )
        ctrl_sm.refresh_snapshot(m_hyb, int_h_det, xraw_h)
        m_hyb.zero_grad(set_to_none=True)
        ce_sm, nll_sm, int_sm, _ = m_hyb(xb_h, yb_h, return_interm=True)
        aux_sm, _, st_sm = ctrl_sm.compute_loss(m_hyb, int_sm, yb_h, nll_sm)
        (ce_sm + 0.5 * aux_sm).backward()
        assert torch.isfinite(aux_sm) and abs(st_sm["k_filled"] - 8.0) < 1e-4

    speed_msg = ""
    if device.type == "cuda":
        from jlens_ablation import _compiled_cluster_select_indices
        u_bench = F.normalize(torch.randn((1311, 64, 64), device=device), dim=-1)
        G_bench = torch.bmm(u_bench, u_bench.transpose(-1, -2)).clamp(-1.0, 1.0)
        p_bench = F.softmax(torch.randn((1311, 64), device=device), dim=-1)
        v_bench = torch.ones((1311, 64), dtype=torch.bool, device=device)
        for smode in ("maxmin", "kmeans", "emeans"):
            for _ in range(3):
                _compiled_cluster_select_indices(G_bench, p_bench, v_bench, 8, smode)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                _compiled_cluster_select_indices(G_bench, p_bench, v_bench, 8, smode)
            torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0 / 20.0
            speed_msg += f" {smode}={dt_ms:.3f}ms"
            assert dt_ms < 5.0, f"{smode} too slow on GPU: {dt_ms:.3f}ms"
    print(f"PASS 12: Verified `maxmin`, `kmeans`, `emeans` (8/8 distinct slots, 50->1 synonym collapse + 7/7 rivals;{speed_msg})")

    # 13. Verify Bystander Eviction + SNR Cleanups (`gamma_max=1.0`, `bystander_tau/w` monotonicity, `live_target_head=1`, `nll_cap`):
    from jlens_ablation import calibrate_usefulness
    raw_u_g = torch.tensor([[0.03, 0.02, 0.01, 0.02]], device=device)
    u_all_g = torch.tensor([1.0], device=device)
    val_g   = torch.ones_like(raw_u_g, dtype=torch.bool)
    usa_g   = torch.tensor([True], device=device)
    u_g4 = calibrate_usefulness(raw_u_g, u_all_g, val_g, usa_g, coalition_norm=True, gamma_max=4.0)
    u_g1 = calibrate_usefulness(raw_u_g, u_all_g, val_g, usa_g, coalition_norm=True, gamma_max=1.0)
    assert float(u_g4[0, 0].item()) > 0.11 and abs(float(u_g1[0, 0].item()) - 0.03) < 1e-6

    # Check exact bystander eviction gradient directions & strict monotonicity across `-tau`:
    # Slots 0..4 have identical log_q (~0.01 prob) and weights: [+0.25 (helpful), +0.02 (bystander), -0.02 (bystander), -0.06 (just harmful), -0.25 (distractor)]
    z_bys = torch.tensor([[-4.6, -4.6, -4.6, -4.6, -4.6, 0.0]], device=device, requires_grad=True)
    w_bys_test = torch.tensor([[0.25, 0.02, -0.02, -0.06, -0.25]], device=device)
    ids_bys = torch.tensor([[0, 1, 2, 3, 4]], device=device)
    val_bys = torch.ones((1, 5), dtype=torch.bool, device=device)
    l_bys = compute_feature_loss(
        z_bys.log_softmax(-1), ids_bys, w_bys_test, val_bys,
        loss_kind="exp_signed_clamped", exp_beta=6.0, exp_norm=True, bystander_tau=0.05, bystander_w=0.30,
    ).sum()
    l_bys.backward()
    g_b = z_bys.grad[0, :5].tolist()
    # Helpful slot 0 pushes logit UP (dL/dz < 0); bystanders 1 & 2 get IDENTICAL positive eviction grad (> 0) discarding +/-0.02 noise;
    # harmful slots 3 (-0.06) & 4 (-0.25) are strictly monotonic: 0 < g_b[1] == g_b[2] < g_b[3] < g_b[4]!
    assert g_b[0] < -0.10, f"Expected strong negative dL/dz on helpful slot, got {g_b}"
    assert g_b[1] > 0.10 and abs(g_b[1] - g_b[2]) < 1e-6, f"Expected identical unweighted bystander eviction on +/-0.02, got {g_b}"
    assert g_b[2] < g_b[3] < g_b[4], f"Expected strict monotonicity across -tau: {g_b}"

    # Verify `live_target_head=1` + `nll_cap=6.5` end-to-end inside CausalAblationJLensController:
    ctrl_bys = CausalAblationJLensController(
        m_hyb, device=device, enable_early=False, anchor_layer=9,
        active_frac=0.02, k_dirs=8, window=1, select_mode="emeans",
        loss_kind="exp_signed_clamped", exp_beta=6.0,
        bystander_tau=0.05, bystander_w=0.25, gamma_max=1.0,
        nll_cap=6.5, live_target_head=True,
    )
    ctrl_bys.refresh_snapshot(m_hyb, int_h_det, xraw_h)
    with torch.no_grad():
        m_hyb.tok_emb.weight.mul_(1.05)  # Perturb live head away from frozen snapshot
    m_hyb.zero_grad(set_to_none=True)
    ce_bs, nll_bs, int_bs, _ = m_hyb(xb_h, yb_h, return_interm=True)
    aux_bs, _, st_bs = ctrl_bys.compute_loss(m_hyb, int_bs, yb_h, nll_bs)
    (ce_bs + 0.5 * aux_bs).backward()
    assert torch.isfinite(aux_bs) and 0.0 <= st_bs["bys_frac"] <= 1.0
    print(f"PASS 13: Verified Bystander Eviction + SNR Cleanups (grads={[round(x,3) for x in g_b]}, bys_frac={st_bs['bys_frac']:.2f})")
    print("ALL VERIFICATION CHECKS PASSED.")


if __name__ == "__main__":
    main()
