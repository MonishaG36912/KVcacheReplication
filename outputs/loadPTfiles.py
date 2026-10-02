"""
Inspect the .pt files saved by Stage 2 (kv_source.pt, kv_target.pt).
Run from inside the outputs/ folder:  python loadPTfiles.py
"""

import torch
import os

FILES = ["kv_source.pt", "kv_target.pt"]

for fname in FILES:
    if not os.path.exists(fname):
        print(f"[MISSING] {fname} — run 01_extract_kv.py first\n")
        continue

    data = torch.load(fname, weights_only=True)
    keys   = data["keys"]
    values = data["values"]

    L, N, H, T, D = keys.shape

    print(f"{'='*55}")
    print(f"  File    : {fname}  ({os.path.getsize(fname)/1e6:.1f} MB on disk)")
    print(f"{'='*55}")
    print(f"  keys   shape : {tuple(keys.shape)}")
    print(f"  values shape : {tuple(values.shape)}")
    print(f"  dtype        : {keys.dtype}")
    print(f"  Layers (L)   : {L}")
    print(f"  Samples (N)  : {N}")
    print(f"  KV heads (H) : {H}")
    print(f"  Seq len (T)  : {T}")
    print(f"  Head dim (D) : {D}")
    print(f"  keys  memory : {keys.nbytes  / 1e6:.1f} MB (in RAM)")
    print(f"  values memory: {values.nbytes / 1e6:.1f} MB (in RAM)")
    print(f"  keys  min/max: {keys.min().item():.4f} / {keys.max().item():.4f}")
    print(f"  keys  mean   : {keys.mean().item():.6f}")
    print()

    sample_layer = 0
    sample_head  = 0
    sample_seq   = 0
    snippet = keys[sample_layer, sample_seq, sample_head, :5, :4]
    print(f"  Sample — layer {sample_layer}, sample {sample_seq}, head {sample_head}, first 5 tokens x 4 dims:")
    print(f"  {snippet.tolist()}")
    print()
