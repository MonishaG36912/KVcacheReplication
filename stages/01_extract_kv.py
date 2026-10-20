"""
Stage 1 & 2 — KV Cache Extraction

Loads Qwen2.5-0.5B (source) and Qwen2.5-1.5B (target), runs the same
calibration chunks through each, and saves the KV cache each model builds.

Calibration data
    N_CHUNKS windows of exactly CHUNK_LEN tokens taken from the WikiText-2
    train split. Every window is real text, so there is no padding. The paper
    calibrates on FineWeb-Edu; its Appendix C shows Wikipedia calibration
    lands within ~1pp of FineWeb-Edu, and WikiText-2 is small enough to
    download in seconds. The test split is kept for Stage 6.

What is captured
    The KV cache that the model itself returns (use_cache=True). Keys in that
    cache are already RoPE-rotated, which is what Stage 3 expects to strip.
    (Earlier versions hooked k_proj, whose output is BEFORE RoPE, so Stage 3
    was rotating keys that had never been rotated.)

Saved outputs (in ./outputs/):
    kv_source.pt  — dict { "keys":   [L_src, N, H, T, D_src]  (RoPE applied),
                           "values": [L_src, N, H, T, D_src],
                           "input_ids": [N, T] }
    kv_target.pt  — same schema for the target model

Tensor dimension legend:
    L = num layers, N = num calibration chunks, H = num KV heads,
    T = tokens per chunk, D = head_dim
"""

import gc
import os
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from kv_utils import SOURCE_MODEL_ID, TARGET_MODEL_ID, load_corpus_ids, make_chunks, cache_to_tensors

torch.set_grad_enabled(False)
torch.manual_seed(42)
DTYPE = torch.bfloat16   # halves RAM vs float32; KV is stored as float32 below

N_CHUNKS = 64     # 64 x 256 = 16,384 calibration tokens per model
CHUNK_LEN = 256
BATCH = 8


def load_model(model_id: str):
    print(f"\n[Load] {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=DTYPE, device_map="cpu", low_cpu_mem_usage=True
    ).eval()
    cfg = model.config
    print(f"  Config  : {cfg.num_hidden_layers} layers, {cfg.num_key_value_heads} KV heads, "
          f"head_dim={cfg.hidden_size // cfg.num_attention_heads}")
    return model


def extract_kv(model, chunks: torch.Tensor) -> dict:
    """Prefill every chunk and copy the resulting cache into (L, N, H, T, D) float32 tensors."""
    cfg = model.config
    L, H = cfg.num_hidden_layers, cfg.num_key_value_heads
    D = cfg.hidden_size // cfg.num_attention_heads
    N, T = chunks.shape
    # Pre-allocate so we never hold two copies of the full KV tensor
    keys = torch.empty(L, N, H, T, D)
    values = torch.empty(L, N, H, T, D)

    t0 = time.time()
    for b in range(0, N, BATCH):
        batch = chunks[b: b + BATCH]
        out = model(input_ids=batch, use_cache=True)
        k, v = cache_to_tensors(out.past_key_values)       # (L, B, H, T, D)
        keys[:, b: b + len(batch)] = k.float()
        values[:, b: b + len(batch)] = v.float()
        print(f"    Processed {min(b + BATCH, N)}/{N} chunks  ({time.time() - t0:.0f}s)")
    return {"keys": keys, "values": values, "input_ids": chunks.clone()}


def main():
    os.makedirs("outputs", exist_ok=True)

    print("=" * 60)
    print("  Stages 1 & 2 — Calibration data and KV extraction")
    print("=" * 60)

    # Both models share the Qwen2.5 tokenizer, so one tokenization serves both
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL_ID)
    tgt_tok = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)
    assert tokenizer.get_vocab() == tgt_tok.get_vocab(), "Source and target tokenizers differ"

    ids = load_corpus_ids(tokenizer, "train")
    chunks = make_chunks(ids, N_CHUNKS, CHUNK_LEN)
    print(f"\n[Data] WikiText-2 train: {ids.numel():,} tokens -> {N_CHUNKS} chunks x {CHUNK_LEN} tokens")

    # One model at a time keeps peak RAM low enough for a 16 GB laptop / WSL
    for model_id, path in [(SOURCE_MODEL_ID, "outputs/kv_source.pt"),
                           (TARGET_MODEL_ID, "outputs/kv_target.pt")]:
        model = load_model(model_id)
        kv = extract_kv(model, chunks)
        del model
        gc.collect()

        K, V = kv["keys"], kv["values"]
        print(f"  keys   shape : {tuple(K.shape)}  — (L, N, H, T, D)")
        print(f"  values shape : {tuple(V.shape)}")
        print(f"  memory       : {(K.nbytes + V.nbytes) / 1e6:.1f} MB")
        torch.save(kv, path)
        print(f"[Saved] {path}")
        del kv, K, V
        gc.collect()

    print("\nDone. Stages 1 & 2 complete.")


if __name__ == "__main__":
    main()
