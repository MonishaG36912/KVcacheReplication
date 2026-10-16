"""
Stage 5 — Closed-form per-head ridge mapper (paper §3.1–3.3)

For every target layer l the input features are the concatenation of ALL KV
heads of its top-k source layers (paper Eq. 5):

    X_K^l = [ K_s^{l1} || K_s^{l2} || ... || K_s^{lk} ]   (RoPE-stripped keys)
    X_V^l = [ V_s^{l1} || V_s^{l2} || ... || V_s^{lk} ]

and each target head h gets its own ridge regression (paper Eq. 3–4):

    W* = (Xc^T Xc + λI)^-1 Xc^T Yc,      b = mean(Y) - mean(X) W*

with Xc, Yc mean-centred and λ = 0.01. All heads of a target layer share the
same X, so solving them together as one multi-output system gives exactly
the per-head solutions (columns of a ridge solve are independent). Keys and
values have separate weights; no parameters are shared between them.

Keys are fit in content space (RoPE stripped on both sides), so the mapper is
position-free. Stage 6 re-applies the TARGET's RoPE when it builds a cache.

Inputs:
    outputs/kv_source_norope.pt, outputs/kv_target_norope.pt   (Stage 3)
    outputs/r2_matrix.pt                                       (Stage 4)

Outputs:
    outputs/mapper_k{k}.pt  — dict with per-target-layer W_K, b_K, W_V, b_V,
                              the selected source layers, and fit diagnostics

Usage:
    python stages/04_fit_mapper.py            # k = 4 (default)
    python stages/04_fit_mapper.py --k 8
    python stages/04_fit_mapper.py --k all
"""

import argparse
import time

import torch
from transformers import AutoConfig

from kv_utils import (SOURCE_MODEL_ID, TARGET_MODEL_ID, source_features, target_rows,
                      r2_score, rope_theta_of)

torch.set_grad_enabled(False)


def fit_ridge(X: torch.Tensor, Y: torch.Tensor, lam: float):
    """
    Closed-form ridge with centring (paper Eq. 4). Covariance is formed in
    float64 because the top-k source layers are correlated and X^T X is
    close to singular. Returns float32 W (d_in, d_out) and b (d_out,).
    """
    X = X.double()
    Y = Y.double()
    x_mean = X.mean(0)
    y_mean = Y.mean(0)
    Xc = X - x_mean
    Yc = Y - y_mean
    A = Xc.T @ Xc
    A.diagonal().add_(lam)
    W = torch.linalg.solve(A, Xc.T @ Yc)
    b = y_mean - x_mean @ W
    return W.float(), b.float()


def predict(X, W, b):
    return torch.addmm(b, X, W)


def parse_k(value: str, L_src: int) -> int:
    k = L_src if value == "all" else int(value)
    if not 1 <= k <= L_src:
        raise ValueError(f"--k must be between 1 and {L_src} (or 'all'), got {value}")
    return k


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", default="4", help="source layers per target layer, or 'all' (default 4)")
    parser.add_argument("--lam", type=float, default=0.01, help="ridge penalty λ (paper: 0.01)")
    parser.add_argument("--val-chunks", type=int, default=8,
                        help="calibration chunks held out to report validation R² (default 8)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Stage 5 — Ridge mapper")
    print("=" * 60)

    print("\n[Load] outputs/kv_source_norope.pt")
    src = torch.load("outputs/kv_source_norope.pt", weights_only=True)
    print("[Load] outputs/kv_target_norope.pt")
    tgt = torch.load("outputs/kv_target_norope.pt", weights_only=True)
    print("[Load] outputs/r2_matrix.pt")
    r2_matrix = torch.load("outputs/r2_matrix.pt", weights_only=True)

    src_kv = {"K": src["keys"].float(), "V": src["values"].float()}
    tgt_kv = {"K": tgt["keys"].float(), "V": tgt["values"].float()}
    del src, tgt

    L_src, N, H_s, T, D_s = src_kv["K"].shape
    L_tgt, _, H_t, _, D_t = tgt_kv["K"].shape
    k = parse_k(args.k, L_src)
    if not 0 < args.val_chunks < N:
        raise ValueError(f"--val-chunks must be between 1 and {N - 1}")

    d_in = k * H_s * D_s
    n_params = 2 * L_tgt * (d_in + 1) * H_t * D_t
    print(f"\n  Source {L_src} layers x {H_s} heads x {D_s} dims, target {L_tgt} x {H_t} x {D_t}")
    print(f"  k = {k} source layers -> {d_in} input features per target layer")
    print(f"  λ = {args.lam},  calibration tokens = {N} x {T} = {N * T:,}")
    print(f"  Mapper parameters (K+V, incl. bias): {n_params / 1e6:.2f}M  "
          f"({n_params * 4 / 1e6:.0f} MB float32)")

    # Top-k source layers per target layer by the Stage 4 selection score
    sources = [torch.topk(r2_matrix[l], k).indices.sort().values for l in range(L_tgt)]

    train = torch.arange(N - args.val_chunks)
    val = torch.arange(N - args.val_chunks, N)

    mapper = {"W_K": [], "b_K": [], "W_V": [], "b_V": [], "sources": sources}
    diag = {f"{kind}_{split}": torch.zeros(L_tgt) for kind in "KV" for split in ("train", "val")}

    print(f"\n[Fit] Train on {len(train)} chunks, validate on {len(val)} held-out chunks,")
    print("      then refit on all chunks for the saved mapper.\n")
    print("  tgt | sources              |  K R² train   val |  V R² train   val")
    print("  ----+----------------------+-------------------+------------------")

    t0 = time.time()
    for l in range(L_tgt):
        idx = sources[l]
        row = {}
        for kind in "KV":
            X_all = source_features(src_kv[kind], idx)            # (N*T, d_in)
            Y_all = target_rows(tgt_kv[kind][l])                  # (N*T, H_t*D_t)
            X_all = X_all.view(N, T, -1)
            Y_all = Y_all.view(N, T, -1)

            X_tr, Y_tr = X_all[train].reshape(-1, d_in), Y_all[train].reshape(-1, H_t * D_t)
            X_va, Y_va = X_all[val].reshape(-1, d_in), Y_all[val].reshape(-1, H_t * D_t)

            W, b = fit_ridge(X_tr, Y_tr, args.lam)
            diag[f"{kind}_train"][l] = r2_score(Y_tr, predict(X_tr, W, b))
            diag[f"{kind}_val"][l] = r2_score(Y_va, predict(X_va, W, b))

            # Final mapper uses every calibration token (as in the paper)
            W, b = fit_ridge(X_all.reshape(-1, d_in), Y_all.reshape(-1, H_t * D_t), args.lam)
            mapper[f"W_{kind}"].append(W)
            mapper[f"b_{kind}"].append(b)

        src_str = ",".join(str(i) for i in idx.tolist())
        if len(src_str) > 20:
            src_str = src_str[:17] + "..."
        print(f"  {l:3d} | {src_str:<20} |   {diag['K_train'][l]:.3f}  {diag['K_val'][l]:6.3f} |"
              f"   {diag['V_train'][l]:.3f}  {diag['V_val'][l]:6.3f}")

    print(f"\n  Fit time: {time.time() - t0:.1f}s")
    print("\n── Summary (R² averaged over target layers and heads) ─────")
    print(f"  Keys   : train {diag['K_train'].mean():.4f}   held-out {diag['K_val'].mean():.4f}")
    print(f"  Values : train {diag['V_train'].mean():.4f}   held-out {diag['V_val'].mean():.4f}")
    worst = diag["V_val"].argmin().item()
    print(f"  Weakest layer (value R², held-out): target layer {worst} = {diag['V_val'][worst]:.4f}")

    src_cfg = AutoConfig.from_pretrained(SOURCE_MODEL_ID)
    tgt_cfg = AutoConfig.from_pretrained(TARGET_MODEL_ID)
    mapper.update({
        "k": k,
        "lambda": args.lam,
        "src_model": SOURCE_MODEL_ID,
        "tgt_model": TARGET_MODEL_ID,
        "src_rope_theta": rope_theta_of(src_cfg),
        "tgt_rope_theta": rope_theta_of(tgt_cfg),
        "tgt_kv_heads": H_t,
        "diagnostics": diag,
    })
    path = f"outputs/mapper_k{k}.pt"
    torch.save(mapper, path)
    print(f"\n[Saved] {path}")
    print("\nDone. Stage 5 complete.")


if __name__ == "__main__":
    main()
