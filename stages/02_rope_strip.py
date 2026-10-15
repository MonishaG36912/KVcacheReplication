"""
Stage 3 — RoPE Stripping (Content-Space Un-rotation)

Reads kv_source.pt and kv_target.pt produced by Stage 2.
Removes the Rotary Position Embedding rotation from Key vectors so
that downstream regression learns content relationships, not
position-dependent ones.

Values (V) pass through unchanged — Qwen2 does not apply RoPE to V.

Outputs:
    outputs/kv_source_norope.pt   — same schema, keys in content space
    outputs/kv_target_norope.pt   — same schema, keys in content space
"""

import torch
from transformers import AutoConfig

SOURCE_MODEL_ID = "Qwen/Qwen2.5-0.5B"
TARGET_MODEL_ID = "Qwen/Qwen2.5-1.5B"


# ── RoPE math ──────────────────────────────────────────────────────────────────

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    The 'rotate_half' used by Qwen2 / LLaMA RoPE:
    splits the last dimension in half and returns [-x2, x1].
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def build_rope_tables(seq_len: int, head_dim: int, rope_theta: float,
                      dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-compute cos and sin tables for positions 0 … seq_len-1.

    Each position vector has the same structure Qwen2 uses internally:
        freq_i = 1 / (rope_theta ^ (2i / head_dim))   for i in [0, head_dim/2)
        angle_i(pos) = pos * freq_i
        embedding = [angle_0, …, angle_{D/2-1},   <- first half
                     angle_0, …, angle_{D/2-1}]   <- repeated second half

    Returns cos and sin both shaped (seq_len, head_dim).
    """
    half = head_dim // 2
    i         = torch.arange(half, dtype=torch.float32)
    inv_freq  = 1.0 / (rope_theta ** (2.0 * i / head_dim))   # (half,)
    positions = torch.arange(seq_len, dtype=torch.float32)    # (T,)
    angles    = torch.outer(positions, inv_freq)               # (T, half)
    emb       = torch.cat([angles, angles], dim=-1)            # (T, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def strip_rope(keys: torch.Tensor, rope_theta: float) -> torch.Tensor:
    """
    Remove RoPE from a keys tensor of shape (L, N, H, T, D).

    Inverse of:  k_rot = k * cos + rotate_half(k) * sin
    Is simply:   k     = k_rot * cos - rotate_half(k_rot) * sin
    (rotation matrix is orthogonal → inverse = transpose = negate the angle)

    Returns tensor of same shape in content space.
    """
    L, N, H, T, D = keys.shape
    cos, sin = build_rope_tables(T, D, rope_theta, dtype=keys.dtype)

    # Expand (T, D) → (1, 1, 1, T, D) for broadcasting across L, N, H
    cos = cos[None, None, None, :, :]
    sin = sin[None, None, None, :, :]

    return keys * cos - rotate_half(keys) * sin


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Stage 3 — RoPE Stripping")
    print("=" * 60)

    # Read rope_theta from cached config (lightweight — no weight download)
    print("\n[Config] Reading rope_theta from model configs …")
    src_cfg = AutoConfig.from_pretrained(SOURCE_MODEL_ID)
    tgt_cfg = AutoConfig.from_pretrained(TARGET_MODEL_ID)
    src_theta = src_cfg.rope_parameters["rope_theta"]
    tgt_theta = tgt_cfg.rope_parameters["rope_theta"]
    print(f"  Source rope_theta : {src_theta}")
    print(f"  Target rope_theta : {tgt_theta}")

    # Load KV tensors saved by Stage 2
    print("\n[Load] outputs/kv_source.pt")
    kv_src = torch.load("outputs/kv_source.pt", weights_only=True)
    print("[Load] outputs/kv_target.pt")
    kv_tgt = torch.load("outputs/kv_target.pt", weights_only=True)

    src_keys = kv_src["keys"]    # (L_src, N, H, T, D_src)
    src_vals = kv_src["values"]
    tgt_keys = kv_tgt["keys"]    # (L_tgt, N, H, T, D_tgt)
    tgt_vals = kv_tgt["values"]

    print(f"\n  Source keys : {tuple(src_keys.shape)}")
    print(f"  Target keys : {tuple(tgt_keys.shape)}")

    # Strip RoPE
    print("\n[Strip] Source keys …")
    src_keys_c = strip_rope(src_keys, src_theta)

    print("[Strip] Target keys …")
    tgt_keys_c = strip_rope(tgt_keys, tgt_theta)

    # ── sanity checks ──────────────────────────────────────────────────────────
    print("\n── Sanity checks ──────────────────────────────────────────")

    # 1. Shape must be unchanged
    assert src_keys_c.shape == src_keys.shape, "Source shape changed!"
    assert tgt_keys_c.shape == tgt_keys.shape, "Target shape changed!"
    print("  Shape preserved          : OK")

    # 2. Keys must actually change (if diff ≈ 0, RoPE was not removed)
    src_diff = (src_keys - src_keys_c).abs().mean().item()
    tgt_diff = (tgt_keys - tgt_keys_c).abs().mean().item()
    print(f"  Mean |Δ| source keys     : {src_diff:.6f}  (non-zero = RoPE removed)")
    print(f"  Mean |Δ| target keys     : {tgt_diff:.6f}  (non-zero = RoPE removed)")

    # 3. Values must be byte-identical (untouched)
    assert torch.equal(src_vals, kv_src["values"]), "Source values were changed!"
    assert torch.equal(tgt_vals, kv_tgt["values"]), "Target values were changed!"
    print("  Values unchanged         : OK")

    # 4. Re-applying RoPE forward to the stripped keys should recover the originals
    def reapply_rope(keys_c, rope_theta):
        L, N, H, T, D = keys_c.shape
        cos, sin = build_rope_tables(T, D, rope_theta, dtype=keys_c.dtype)
        cos = cos[None, None, None, :, :]
        sin = sin[None, None, None, :, :]
        return keys_c * cos + rotate_half(keys_c) * sin

    src_recovered = reapply_rope(src_keys_c, src_theta)
    recovery_err  = (src_recovered - src_keys).abs().max().item()
    print(f"  Round-trip max error     : {recovery_err:.2e}  (should be < 1e-4)")

    # ── save ──────────────────────────────────────────────────────────────────
    torch.save({"keys": src_keys_c, "values": src_vals}, "outputs/kv_source_norope.pt")
    torch.save({"keys": tgt_keys_c, "values": tgt_vals}, "outputs/kv_target_norope.pt")

    print("\n[Saved] outputs/kv_source_norope.pt")
    print("[Saved] outputs/kv_target_norope.pt")
    print("\nDone. Stage 3 complete.")


if __name__ == "__main__":
    main()
