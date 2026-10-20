"""
Stage 6 — Benchmark the mapped KV cache inside the real target model

Evaluation data is the WikiText-2 TEST split (Stage 1 calibrated on train).

1. Transfer quality — prefix-conditioned perplexity (paper Appendix D)
   Each chunk is split into a prefix (P tokens) and a continuation (C tokens).
   The target model scores the SAME continuation tokens under different
   prefix caches, so only the quality of the prefix KV differs:

     target standalone   target prefills the prefix itself        (ceiling)
     GT cache round-trip target's own cache -> strip RoPE ->      (plumbing check,
                         re-apply RoPE -> rebuilt cache            should equal ceiling)
     transfer            source prefills the prefix, the Stage 5  (the method)
                         mapper converts its cache to the target
     target, no prefix   target sees only the continuation         (floor)
     source standalone   the small model keeps going on its own    (reference)

   Reported: perplexity, top-1 next-token agreement with the standalone
   target, mean KL(standalone || mode), and "prefix benefit retained":
       (PPL_no_prefix - PPL_mode) / (PPL_no_prefix - PPL_standalone)
   which is 100% for a perfect cache and 0% for a cache worth nothing.

2. Held-out KV reconstruction — R² of mapped vs. ground-truth target KV on
   the test prefixes (content space), per target layer.

3. Prefill latency (paper §4.7) — target re-prefill of the transformer body
   (no LM head) vs. applying the mapper to an existing source cache.

Usage:
    python stages/05_benchmark.py                 # uses outputs/mapper_k4.pt
    python stages/05_benchmark.py --k 8 --n-chunks 32
    python stages/05_benchmark.py --skip-latency

Outputs:
    outputs/benchmark_k{k}.json — all numbers printed below
"""

import argparse
import json
import math
import statistics
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_utils import (SOURCE_MODEL_ID, TARGET_MODEL_ID, load_corpus_ids, make_chunks,
                      cache_to_tensors, tensors_to_cache, strip_rope, apply_rope,
                      apply_mapper, target_rows, r2_score)

torch.set_grad_enabled(False)
DTYPE = torch.bfloat16

MODES = ["target standalone", "GT cache round-trip", "transfer", "target, no prefix", "source standalone"]


def load_model(model_id):
    print(f"[Load] {model_id}")
    return AutoModelForCausalLM.from_pretrained(
        model_id, dtype=DTYPE, device_map="cpu", low_cpu_mem_usage=True).eval()


def continuation_logprobs(model, cont_ids, cache=None):
    """Log-probs for cont_ids[1:] given the prefix cache (or no prefix). Shape (C-1, V)."""
    logits = model(input_ids=cont_ids, past_key_values=cache, use_cache=cache is not None).logits
    return F.log_softmax(logits[0, :-1].float(), dim=-1)


def transfer_cache(src_keys, src_vals, mapper, tgt_config):
    """Source cache tensors (L_s,1,H,T,D) -> target DynamicCache via the ridge mapper."""
    k_c = strip_rope(src_keys.float(), mapper["src_rope_theta"])
    tk_c, tv = apply_mapper(mapper, k_c, src_vals)
    tk = apply_rope(tk_c, mapper["tgt_rope_theta"])
    return tensors_to_cache(tk.to(DTYPE), tv.to(DTYPE), tgt_config), tk_c, tv


def roundtrip_cache(tgt_keys, tgt_vals, theta, tgt_config):
    """Target's own cache pushed through the same strip / re-rotate / rebuild path."""
    k = apply_rope(strip_rope(tgt_keys.float(), theta), theta)
    return tensors_to_cache(k.to(DTYPE), tgt_vals, tgt_config)


def evaluate_quality(src_model, tgt_model, mapper, chunks, P):
    nll = {m: 0.0 for m in MODES}
    agree = {m: 0 for m in MODES}
    kl = {m: 0.0 for m in MODES}
    n_tok = 0
    kv_true = {"K": [], "V": []}
    kv_pred = {"K": [], "V": []}
    tgt_theta = mapper["tgt_rope_theta"]

    t0 = time.time()
    for i, ids in enumerate(chunks):
        ids = ids.unsqueeze(0)
        prefix, cont = ids[:, :P], ids[:, P:]
        labels = cont[0, 1:]

        # Ground truth: target prefills the prefix itself
        tgt_k, tgt_v = cache_to_tensors(tgt_model(input_ids=prefix, use_cache=True).past_key_values)
        # Source prefill (the cache we want to reuse)
        src_k, src_v = cache_to_tensors(src_model(input_ids=prefix, use_cache=True).past_key_values)

        mapped_cache, tk_c, tv = transfer_cache(src_k, src_v, mapper, tgt_model.config)

        lp = {
            "target standalone": continuation_logprobs(tgt_model, cont, tensors_to_cache(tgt_k, tgt_v, tgt_model.config)),
            "GT cache round-trip": continuation_logprobs(tgt_model, cont, roundtrip_cache(tgt_k, tgt_v, tgt_theta, tgt_model.config)),
            "transfer": continuation_logprobs(tgt_model, cont, mapped_cache),
            "target, no prefix": continuation_logprobs(tgt_model, cont),
            "source standalone": continuation_logprobs(src_model, cont, tensors_to_cache(src_k, src_v, src_model.config)),
        }
        ref = lp["target standalone"]
        ref_top1 = ref.argmax(-1)
        for m in MODES:
            nll[m] -= lp[m].gather(1, labels[:, None]).sum().item()
            agree[m] += (lp[m].argmax(-1) == ref_top1).sum().item()
            kl[m] += F.kl_div(lp[m], ref, log_target=True, reduction="sum").item()
        n_tok += labels.numel()

        # Held-out reconstruction, content space (target keys stripped of target RoPE)
        kv_true["K"].append(strip_rope(tgt_k.float(), tgt_theta)[:, 0])
        kv_true["V"].append(tgt_v.float()[:, 0])
        kv_pred["K"].append(tk_c[:, 0])
        kv_pred["V"].append(tv[:, 0])

        ppl_t = math.exp(nll["transfer"] / n_tok)
        ppl_s = math.exp(nll["target standalone"] / n_tok)
        print(f"    chunk {i + 1:3d}/{len(chunks)}  running PPL standalone {ppl_s:7.2f}  "
              f"transfer {ppl_t:7.2f}  ({time.time() - t0:.0f}s)")

    results = {}
    for m in MODES:
        results[m] = {
            "ppl": math.exp(nll[m] / n_tok),
            "top1_agreement": agree[m] / n_tok,
            "kl_from_standalone": kl[m] / n_tok,
        }
    floor, ceil = results["target, no prefix"]["ppl"], results["target standalone"]["ppl"]
    for m in MODES:
        results[m]["prefix_benefit_retained"] = (floor - results[m]["ppl"]) / (floor - ceil) if floor != ceil else float("nan")

    # Per-layer R²: stack chunks along a new N axis -> (L, N, H, T, D)
    r2 = {}
    for kind in "KV":
        true = torch.stack(kv_true[kind], dim=1)
        pred = torch.stack(kv_pred[kind], dim=1)
        r2[kind] = [r2_score(target_rows(true[l]), target_rows(pred[l])) for l in range(true.shape[0])]
    return results, r2, n_tok


def timed(fn, trials):
    fn()  # warm-up
    times = []
    for _ in range(trials):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def evaluate_latency(src_model, tgt_model, mapper, ids, lengths, trials):
    rows = []
    for L in lengths:
        x = ids[:L].unsqueeze(0)
        src_k, src_v = cache_to_tensors(src_model(input_ids=x, use_cache=True).past_key_values)
        re_prefill = timed(lambda: tgt_model.model(input_ids=x, use_cache=True), trials)
        mapper_ms = timed(lambda: transfer_cache(src_k, src_v, mapper, tgt_model.config), trials)
        src_prefill = timed(lambda: src_model.model(input_ids=x, use_cache=True), trials)
        rows.append({"seq_len": L, "target_reprefill_ms": re_prefill, "mapper_ms": mapper_ms,
                     "source_prefill_ms": src_prefill, "speedup": re_prefill / mapper_ms})
        print(f"    {L:5d} tokens  re-prefill {re_prefill:8.1f} ms   mapper {mapper_ms:7.1f} ms   "
              f"speedup {re_prefill / mapper_ms:5.1f}x   (source prefill {src_prefill:7.1f} ms)")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=4,
                        help="which Stage 5 mapper to load, outputs/mapper_k{k}.pt (k=all is saved as k=24)")
    parser.add_argument("--n-chunks", type=int, default=16, help="test chunks to evaluate (default 16)")
    parser.add_argument("--prefix", type=int, default=256, help="prefix tokens whose KV is transferred")
    parser.add_argument("--cont", type=int, default=128, help="continuation tokens scored")
    parser.add_argument("--latency-lens", default="64,256,1024", help="comma-separated sequence lengths")
    parser.add_argument("--trials", type=int, default=3, help="timed trials per latency cell (median)")
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  Stage 6 — Benchmark")
    print("=" * 60)

    mapper_path = f"outputs/mapper_k{args.k}.pt"
    print(f"[Load] {mapper_path}")
    mapper = torch.load(mapper_path, weights_only=True)
    print(f"  k = {mapper['k']}, λ = {mapper['lambda']}")

    tok = AutoTokenizer.from_pretrained(SOURCE_MODEL_ID)
    test_ids = load_corpus_ids(tok, "test")
    chunks = make_chunks(test_ids, args.n_chunks, args.prefix + args.cont)
    print(f"[Data] WikiText-2 test: {test_ids.numel():,} tokens -> {args.n_chunks} chunks "
          f"of {args.prefix} prefix + {args.cont} continuation tokens")

    src_model = load_model(SOURCE_MODEL_ID)
    tgt_model = load_model(TARGET_MODEL_ID)

    print("\n[1] Transfer quality (prefix-conditioned perplexity)")
    results, r2, n_tok = evaluate_quality(src_model, tgt_model, mapper, chunks, args.prefix)

    print(f"\n  Scored tokens: {n_tok:,}  (same tokens in every row)\n")
    print(f"  {'mode':<22} {'PPL':>8} {'benefit kept':>13} {'top-1 agree':>12} {'KL':>8}")
    for m in MODES:
        r = results[m]
        print(f"  {m:<22} {r['ppl']:8.2f} {100 * r['prefix_benefit_retained']:12.1f}% "
              f"{100 * r['top1_agreement']:11.1f}% {r['kl_from_standalone']:8.4f}")

    print("\n[2] Held-out KV reconstruction R² (test prefixes, content space)")
    print("  tgt layer:  " + " ".join(f"{l:5d}" for l in range(len(r2["K"]))))
    for kind in "KV":
        print(f"  {kind} R²    :  " + " ".join(f"{v:5.2f}" for v in r2[kind]))
    print(f"  mean K R² = {sum(r2['K']) / len(r2['K']):.4f},  mean V R² = {sum(r2['V']) / len(r2['V']):.4f}")

    latency = []
    if not args.skip_latency:
        print(f"\n[3] Prefill latency (median of {args.trials}, CPU, {torch.get_num_threads()} threads)")
        lengths = [int(x) for x in args.latency_lens.split(",")]
        latency = evaluate_latency(src_model, tgt_model, mapper, test_ids, lengths, args.trials)

    out = {
        "k": mapper["k"], "lambda": mapper["lambda"],
        "n_chunks": args.n_chunks, "prefix": args.prefix, "cont": args.cont, "scored_tokens": n_tok,
        "quality": results,
        "kv_r2_heldout": {"keys": r2["K"], "values": r2["V"]},
        "latency": latency,
    }
    path = f"outputs/benchmark_k{mapper['k']}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Saved] {path}")
    print("\nDone. Stage 6 complete.")


if __name__ == "__main__":
    main()
