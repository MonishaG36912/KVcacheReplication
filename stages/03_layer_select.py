"""
Stage 4 — Top-k Source Layer Selection

For every (source layer, target layer, head) triple, fits a quick OLS
regression on the RoPE-stripped keys and on the values, and records R² —
how well the source layer linearly predicts the target layer.

Following paper §3.2, the selection score is the head-averaged R² averaged
over RoPE-stripped keys and values. That gives a (L_tgt × L_src) score
matrix, from which the top-k source layers are picked for each target layer.

Outputs:
    outputs/r2_matrix.pt      — float tensor (L_tgt, L_src), mean of key and value R²
    outputs/r2_kv.pt          — dict {"keys": (L_tgt, L_src), "values": (L_tgt, L_src)}
    outputs/top_k_mapping.pt  — long tensor  (L_tgt, k) of selected source indices
"""

import torch

K        = 4      # number of source layers to select per target layer
LAMBDA   = 0.01   # ridge penalty used when solving OLS (keeps it stable)


# ── OLS utilities ──────────────────────────────────────────────────────────────

def ols_r2(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Fit ridge-regularised OLS  W* = (X'X + λI)^{-1} X'Y  and return R².

    X : (M, D_in)   — feature matrix (flattened samples × tokens)
    Y : (M, D_out)  — target matrix

    R² is averaged across all D_out output dimensions.
    A value near 1.0 means source nearly perfectly predicts target;
    near 0.0 means no linear relationship.
    """
    M, D_in = X.shape

    # Centre both matrices (removes intercept from the regression problem)
    X_c = X - X.mean(dim=0, keepdim=True)
    Y_c = Y - Y.mean(dim=0, keepdim=True)

    # Gram matrix with ridge penalty → (D_in, D_in)
    A = X_c.T @ X_c + LAMBDA * torch.eye(D_in, dtype=X.dtype)
    B = X_c.T @ Y_c  # (D_in, D_out)

    # Solve A W = B  →  W* = A^{-1} B
    W = torch.linalg.solve(A, B)          # (D_in, D_out)

    # Predictions and residuals
    Y_hat   = X_c @ W                     # (M, D_out)
    ss_res  = ((Y_c - Y_hat) ** 2).sum(dim=0)   # (D_out,)
    ss_tot  = (Y_c ** 2).sum(dim=0)              # (D_out,)

    # Per-output R², clipped to [0,1] then averaged
    r2_per_dim = 1.0 - ss_res / (ss_tot + 1e-12)
    r2_per_dim = r2_per_dim.clamp(0.0, 1.0)
    return r2_per_dim.mean().item()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Stage 4 — Top-k Source Layer Selection")
    print("=" * 60)

    # Load RoPE-stripped keys
    print("\n[Load] outputs/kv_source_norope.pt")
    src = torch.load("outputs/kv_source_norope.pt", weights_only=True)
    print("[Load] outputs/kv_target_norope.pt")
    tgt = torch.load("outputs/kv_target_norope.pt", weights_only=True)

    src_kv = {"keys": src["keys"].float(), "values": src["values"].float()}  # (L_src, N, H, T, D_src)
    tgt_kv = {"keys": tgt["keys"].float(), "values": tgt["values"].float()}  # (L_tgt, N, H, T, D_tgt)

    L_src, N, H, T, D_src = src_kv["keys"].shape
    L_tgt, _, _, _, D_tgt = tgt_kv["keys"].shape

    print(f"\n  Source : {L_src} layers, head_dim={D_src}")
    print(f"  Target : {L_tgt} layers, head_dim={D_tgt}")
    print(f"  Heads  : {H},  Samples: {N},  Seq len: {T}")
    print(f"  Regressions to run: 2 (K,V) × {L_tgt} × {L_src} × {H} = {2*L_tgt*L_src*H}")

    # ── compute R² matrices ───────────────────────────────────────────────────
    # r2[kind][l_tgt, l_src] = mean R² across heads
    r2 = {kind: torch.zeros(L_tgt, L_src) for kind in ("keys", "values")}

    total = L_tgt * L_src
    done  = 0

    print(f"\n[R²] Computing …  (0/{total})", end="", flush=True)

    for l_t in range(L_tgt):
        for l_s in range(L_src):
            for kind in ("keys", "values"):
                head_r2 = []
                for h in range(H):
                    # Flatten (N, T, D) → (N*T, D)
                    X = src_kv[kind][l_s, :, h, :, :].reshape(N * T, D_src)
                    Y = tgt_kv[kind][l_t, :, h, :, :].reshape(N * T, D_tgt)
                    head_r2.append(ols_r2(X, Y))
                r2[kind][l_t, l_s] = sum(head_r2) / len(head_r2)
            done += 1
            if done % 50 == 0 or done == total:
                print(f"\r[R²] Computing …  ({done}/{total})", end="", flush=True)

    print()   # newline after progress

    # Paper §3.2: rank source layers by R² averaged over stripped keys and values
    r2_matrix = (r2["keys"] + r2["values"]) / 2

    # ── top-k selection ───────────────────────────────────────────────────────
    # For each target layer, pick k source layers with highest R²
    top_k_vals, top_k_idx = torch.topk(r2_matrix, k=K, dim=1)
    # top_k_idx : (L_tgt, K) — source layer indices, sorted best-first

    # ── print R² heatmap (relative scale so variation is visible) ────────────
    # Scale: . = lowest, + = low-mid, o = mid-high, # = highest
    # Thresholds are percentiles of the actual data, not fixed values
    r2_min = r2_matrix.min().item()
    r2_max = r2_matrix.max().item()
    r2_range = r2_max - r2_min + 1e-9

    CHARS = [".","+"  , "o", "#"]   # ASCII, renders in every terminal

    print("\n── R2 Heatmap  (rows=target layers 0-27, cols=source layers 0-23) ──")
    print("   Scale: . = low R2   + = mid-low   o = mid-high   # = high")
    print(f"   R2 range in this run: {r2_min:.3f} (.) to {r2_max:.3f} (#)")
    print()
    print("       src: 0         1         2    ")
    print("            0123456789012345678901234")

    for l_t in range(L_tgt):
        row = r2_matrix[l_t]
        bar = ""
        for v in row:
            bucket = int((v.item() - r2_min) / r2_range * (len(CHARS) - 1) + 0.5)
            bucket = max(0, min(len(CHARS) - 1, bucket))
            bar += CHARS[bucket]
        top_str = ",".join(str(i.item()) for i in top_k_idx[l_t])
        best_r2 = top_k_vals[l_t, 0].item()
        print(f"  tgt {l_t:2d} |{bar}| top=[{top_str}] R2={best_r2:.3f}")

    # ── summary statistics ────────────────────────────────────────────────────
    print(f"\n  Mean key R²      : {r2['keys'].mean().item():.4f}")
    print(f"  Mean value R²    : {r2['values'].mean().item():.4f}")
    print(f"  Overall mean R²  : {r2_matrix.mean().item():.4f}  (selection score, K and V averaged)")
    print(f"  Overall max R²   : {r2_matrix.max().item():.4f}")
    print(f"  Overall min R²   : {r2_matrix.min().item():.4f}")

    # Which source layer is most universally useful?
    src_popularity = torch.bincount(top_k_idx.flatten(), minlength=L_src)
    most_popular   = src_popularity.argmax().item()
    print(f"  Most-selected source layer : {most_popular} "
          f"(chosen for {src_popularity[most_popular].item()}/{L_tgt} target layers)")

    # ── save ──────────────────────────────────────────────────────────────────
    torch.save(r2_matrix,  "outputs/r2_matrix.pt")
    torch.save(r2,         "outputs/r2_kv.pt")
    torch.save(top_k_idx,  "outputs/top_k_mapping.pt")

    print("\n[Saved] outputs/r2_matrix.pt")
    print("[Saved] outputs/r2_kv.pt")
    print("[Saved] outputs/top_k_mapping.pt")
    print("\nDone. Stage 4 complete.")


if __name__ == "__main__":
    main()
