"""
Axes demo — slice a real KV cache along each of its five axes.

Uses only the 0.5B model, so it runs in about 20 seconds:
    python stages/axes_demo.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_utils import SOURCE_MODEL_ID, cache_to_tensors, strip_rope, rope_theta_of

torch.set_grad_enabled(False)

PROMPT = ("The Eiffel Tower is in Paris, and the Colosseum is in Rome. "
          "Paris is the capital of France, and Rome is the capital of")


def vec(x, n=4):
    return "[" + ", ".join(f"{v:6.2f}" for v in x[:n].tolist()) + ", ...]"


def main():
    tok = AutoTokenizer.from_pretrained(SOURCE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        SOURCE_MODEL_ID, dtype=torch.float32, attn_implementation="eager").eval()

    ids = tok(PROMPT, return_tensors="pt").input_ids
    pieces = [tok.decode(t) for t in ids[0]]
    prefix = ids[:, :-1]
    T = prefix.shape[1]

    print("\n=== AXIS 4: TOKENS — the prompt split into pieces ===")
    for i in range(0, len(pieces), 8):
        print("  " + "  ".join(f"{j:2d}:{pieces[j]!r}" for j in range(i, min(i + 8, len(pieces)))))
    print(f"  Tokens 0..{T - 1} go into the cache. Token {T} ({pieces[T]!r}) is fed live.")
    paris = pieces.index(" Paris")
    rome = pieces.index(" Rome")

    out = model(input_ids=prefix, use_cache=True)
    keys, values = cache_to_tensors(out.past_key_values)
    print(f"\n  keys shape = {tuple(keys.shape)}  (layers, batch, kv-heads, tokens, head_dim)")

    print("\n=== Moving along TOKENS (layer 15, head 0): one key per token ===")
    for t in [0, paris, rome, T - 1]:
        print(f"  keys[15, 0, 0, {t:2d}] token {pieces[t]!r:<12} {vec(keys[15, 0, 0, t])}")

    print("\n=== Moving along LAYERS (token ' Paris', head 0): the same token, re-described at each stage ===")
    for l in [0, 5, 10, 15, 20, 23]:
        k = keys[l, 0, 0, paris]
        print(f"  keys[{l:2d}, 0, 0, {paris}]  {vec(k)}   length {k.norm():6.1f}")

    print("\n=== Moving along KV HEADS (layer 15, token ' Paris'): two parallel views ===")
    for h in range(keys.shape[2]):
        print(f"  keys[15, 0, {h}, {paris}]  {vec(keys[15, 0, h, paris])}")

    print("\n=== Moving along HEAD_DIM: all 64 numbers of ONE key (layer 15, head 0, ' Paris') ===")
    k = keys[15, 0, 0, paris]
    for i in range(0, 64, 16):
        print("  " + " ".join(f"{v:6.2f}" for v in k[i:i + 16].tolist()))
    print(f"  and its VALUE, first numbers: {vec(values[15, 0, 0, paris], 8)}")

    print("\n=== Same word, two places in the prompt (does context change the key?) ===")
    print("  RoPE is stripped first, so position is removed and only content is compared.")
    content = strip_rope(keys, rope_theta_of(model.config))
    first_paris = paris
    second_paris = [i for i, p in enumerate(pieces) if p == " Paris"][-1]
    for l in [0, 5, 15, 23]:
        a, b = content[l, 0, 0, first_paris], content[l, 0, 0, second_paris]
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        print(f"  layer {l:2d}: token {first_paris} {pieces[first_paris]!r} vs token {second_paris} "
              f"{pieces[second_paris]!r}  cosine {cos:.2f}")

    print("\n=== AXIS 2: BATCH — two prompts processed side by side ===")
    batch = tok([PROMPT, PROMPT.replace("France", "Italy")], return_tensors="pt", padding=True)
    lens = batch.attention_mask.sum(1).tolist()
    if lens[0] == lens[1]:
        bk, _ = cache_to_tensors(model(input_ids=batch.input_ids[:, :-1], use_cache=True).past_key_values)
        print(f"  keys shape = {tuple(bk.shape)}   <- batch axis is now 2")
        france = pieces.index(" France")
        for t in [rome, france, T - 1]:
            for b in range(2):
                name = tok.decode(batch.input_ids[b, t])
                print(f"  keys[15, {b}, 0, {t:2d}] prompt {b}, token {name!r:<10} {vec(bk[15, b, 0, t])}")
    else:
        print(f"  (skipped: the two prompts tokenize to different lengths {lens})")

    print("\n=== How the cache is USED: where does ' of' look? ===")
    att = model(input_ids=ids, output_attentions=True).attentions   # per layer (1, q_heads, T+1, T+1)
    for l in [3, 15, 23]:
        w = att[l][0, :, -1, :].mean(0)          # last token's attention, averaged over query heads
        top = w.topk(5)
        print(f"  layer {l:2d}: " + ",  ".join(f"{i}:{pieces[i]!r} {100 * p:.0f}%"
                                                for p, i in zip(top.values.tolist(), top.indices.tolist())))
    print("  ' of' compares its query with every stored KEY, then mixes those tokens' VALUES by these weights.")


if __name__ == "__main__":
    main()
