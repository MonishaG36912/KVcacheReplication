"""
Walkthrough — follow ONE prompt through the whole transfer, step by step.

Needs outputs/mapper_k4.pt from Stage 5. Takes about a minute on CPU.

    python stages/walkthrough.py
    python stages/walkthrough.py --prompt "Your own text here and the model will continue"
"""

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_utils import (SOURCE_MODEL_ID, TARGET_MODEL_ID, cache_to_tensors, tensors_to_cache,
                      strip_rope, apply_rope, apply_mapper)

torch.set_grad_enabled(False)
DTYPE = torch.bfloat16


def top5(model, tok, last_token, cache=None):
    logits = model(input_ids=last_token, past_key_values=cache, use_cache=cache is not None).logits[0, -1]
    probs = F.softmax(logits.float(), dim=-1)
    p, i = probs.topk(5)
    return ",  ".join(f"{tok.decode(t)!r} {100 * q:.0f}%" for t, q in zip(i.tolist(), p.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The Eiffel Tower is in Paris, and the Colosseum is in Rome. "
                                        "Paris is the capital of France, and Rome is the capital of")
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SOURCE_MODEL_ID)
    mapper = torch.load(f"outputs/mapper_k{args.k}.pt", weights_only=True)
    src = AutoModelForCausalLM.from_pretrained(SOURCE_MODEL_ID, dtype=DTYPE).eval()
    tgt = AutoModelForCausalLM.from_pretrained(TARGET_MODEL_ID, dtype=DTYPE).eval()

    ids = tok(args.prompt, return_tensors="pt").input_ids
    prefix, last = ids[:, :-1], ids[:, -1:]
    print(f"\nPrompt: {args.prompt!r}")
    print(f"  {ids.shape[1]} tokens. The first {prefix.shape[1]} are the 'prefix' whose KV cache we transfer;")
    print(f"  the last token {tok.decode(last[0])!r} is fed live so the model predicts what comes next.\n")

    print("STEP 1  The small model reads the prefix (this is the prefill we want to reuse)")
    sk, sv = cache_to_tensors(src(input_ids=prefix, use_cache=True).past_key_values)
    print(f"  source cache keys   : {tuple(sk.shape)}  = (layers, batch, kv-heads, tokens, head_dim)")

    print("\nSTEP 2  Remove RoPE from the source keys, so only 'content' is left")
    sk_c = strip_rope(sk.float(), mapper["src_rope_theta"])
    print(f"  token 3, source layer 15, head 0, first 4 dims  rotated: {sk[15, 0, 0, 3, :4].float().tolist()}")
    print(f"                                                  content: {[round(x, 3) for x in sk_c[15, 0, 0, 3, :4].tolist()]}")

    print("\nSTEP 3  The ridge mapper turns source content KV into target content KV")
    tk_c, tv = apply_mapper(mapper, sk_c, sv)
    print(f"  mapped target keys  : {tuple(tk_c.shape)}  (28 layers, head_dim 128)")
    print(f"  target layer 10 is built from source layers {mapper['sources'][10].tolist()}")

    print("\nSTEP 4  Compare with what the big model would have computed itself")
    gk, gv = cache_to_tensors(tgt(input_ids=prefix, use_cache=True).past_key_values)
    gk_c = strip_rope(gk.float(), mapper["tgt_rope_theta"])
    for name, a, b in [("keys", tk_c, gk_c), ("values", tv, gv.float())]:
        cos = F.cosine_similarity(a.flatten(3), b.flatten(3), dim=-1).mean().item()
        print(f"  mean cosine(mapped, real) for {name:<6}: {cos:.3f}   (1.0 = identical direction)")

    print("\nSTEP 5  Re-apply the TARGET's RoPE, build a cache, and let the big model continue")
    mapped_cache = tensors_to_cache(apply_rope(tk_c, mapper["tgt_rope_theta"]).to(DTYPE), tv.to(DTYPE), tgt.config)

    print("\nNext-token predictions (top 5):")
    print(f"  1.5B, own prefill     : {top5(tgt, tok, last, tensors_to_cache(gk, gv, tgt.config))}")
    print(f"  1.5B, TRANSFERRED KV  : {top5(tgt, tok, last, mapped_cache)}")
    print(f"  1.5B, no prefix at all: {top5(tgt, tok, last)}")
    print(f"  0.5B, own prefill     : {top5(src, tok, last, tensors_to_cache(sk, sv, src.config))}")


if __name__ == "__main__":
    main()
