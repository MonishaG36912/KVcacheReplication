# Technical Report — Cross-Model KV Cache Transfer Replication

**Author:** Monisha  
**Hardware:** AMD Ryzen 7 7735U · 16 GB RAM · CPU-only  
**Models:** `Qwen/Qwen2.5-0.5B` → `Qwen/Qwen2.5-1.5B`  
**Reference Paper:** Heo et al., NVIDIA, 2026

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Background: What is a KV Cache?](#2-background-what-is-a-kv-cache)
3. [Stage 1 & 2 — Model Loading & KV Extraction](#3-stage-1--2--model-loading--kv-extraction)
4. [Stage 3 — RoPE Stripping](#4-stage-3--rope-stripping)
5. [Stage 4 — Top-k Source Layer Selection](#5-stage-4--top-k-source-layer-selection)
6. [Results Summary](#6-results-summary)
7. [Stage 5 — Ridge Mapper](#7-stage-5--ridge-mapper)
8. [Stage 6 — Benchmark](#8-stage-6--benchmark)
9. [What Comes Next (per the paper)](#9-what-comes-next-per-the-paper)
10. [Sources](#10-sources)

---

## 1. Project Goal

When two language models in the same family (one small, one large) process the same text, they produce internal Key-Value (KV) tensors at every attention layer. The paper asks: **can we train a linear function that converts the small model's KV tensors into the large model's expected KV format**, so the large model never has to re-read the prompt?

If yes, this skips the **prefill** step entirely when switching models — the most expensive part of serving long prompts.

The pipeline has six stages:

```
Load Models → Extract KV → Strip RoPE → Select Layers → Fit Mapper → Benchmark
   Stage 1        Stage 2     Stage 3       Stage 4       Stage 5     Stage 6
```

All six stages are complete. This report covers them in full.

> **Phase 2 correction.** The Phase 1 extraction hooked `k_proj`, whose output is the key *before* RoPE is applied. Stage 3 therefore "un-rotated" keys that had never been rotated, and the 15 calibration sentences were ~90% padding. Phase 2 reads K/V from the model's own cache and calibrates on 64 unpadded WikiText-2 chunks. All numbers below come from the corrected pipeline; the Phase 1 R² of 0.777 should not be used.

---

## 2. Background: What is a KV Cache?

### Transformer Attention

Every transformer layer computes **self-attention**. For an input sequence of T tokens, each with hidden dimension `d_model`, the layer:

1. Projects the input X into three matrices:

```
Q = X · W_Q    (Queries)   shape: (T, d_head)
K = X · W_K    (Keys)      shape: (T, d_head)
V = X · W_V    (Values)    shape: (T, d_head)
```

2. Computes attention scores and weighted values:

```
Attention(Q, K, V) = softmax( Q · K^T / sqrt(d_head) ) · V
```

The softmax decides which tokens each position should "pay attention to." The result is a weighted mixture of value vectors.

### Why Cache K and V?

During **generation** (producing tokens one by one), the model re-uses the Keys and Values from all previous tokens. Storing K and V avoids re-computing them at every new token — this is the **KV cache**.

**Prefill** is the forward pass that computes and stores K, V for the entire input prompt before generation begins. For a 10,000-token prompt on a 70B model, this takes several seconds.

### The Transfer Problem

If we switch from a small model S to a large model T mid-conversation:

- S has already computed its KV cache C_S (shape dependent on S's architecture)
- T needs its own KV cache C_T (different shape, different internal representation)
- Without transfer: T re-reads the full prompt → full prefill cost
- With transfer: a mapper f converts C_S → Ĉ_T → T decodes using Ĉ_T directly

The paper shows this mapper can be a simple **linear function** fitted with ridge regression.

### Tensor Shapes in This Project

After extraction (Stage 2), KV tensors are shaped:

```
(L, N, H, T, D)

L = number of transformer layers
N = number of calibration samples
H = number of KV attention heads
T = sequence length (tokens)
D = head dimension
```

| Model | L | H | D | Full shape |
|---|---|---|---|---|
| Qwen2.5-0.5B (source) | 24 | 2 | 64 | (24, 64, 2, 256, 64) |
| Qwen2.5-1.5B (target) | 28 | 2 | 128 | (28, 64, 2, 256, 128) |

---

## 3. Stage 1 & 2 — Model Loading & KV Extraction

**Script:** `stages/01_extract_kv.py`

### 3.1 What Hugging Face Provides

`AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")` downloads the model weights (~1 GB) from huggingface.co and builds the full transformer in Python. The internal structure for Qwen2.5 is:

```
model.model.layers[0]          ← transformer layer 0
    .self_attn                 ← attention module
        .q_proj                ← Linear: d_model → n_heads * d_head
        .k_proj                ← Linear: d_model → n_kv_heads * d_head
        .v_proj                ← Linear: d_model → n_kv_heads * d_head
        .o_proj                ← Linear: n_heads * d_head → d_model
model.model.layers[1]
    ...
model.model.layers[23]         ← last layer (0.5B has 24 layers)
```

Qwen2.5 uses **Grouped Query Attention (GQA)**: instead of one K/V head per Q head, multiple Q heads share a single K/V head. The 0.5B model has 14 Q heads but only 2 KV heads.

### 3.2 Reading the Cache the Model Builds

Inside `Qwen2Attention.forward` the order is:

```
k = k_proj(x)                     ← content key, NO position information yet
q, k = apply_rotary_pos_emb(q, k) ← RoPE rotates k by its position
cache.update(k, v)                ← the cache stores the ROTATED key
```

Phase 1 hooked `k_proj`, so it saved the un-rotated key. Phase 2 simply runs the model with `use_cache=True` and copies the returned `DynamicCache`, which holds exactly what the model stores and later attends to:

```python
out = model(input_ids=batch, use_cache=True)
keys   = torch.stack([layer.keys   for layer in out.past_key_values.layers])  # (L, B, H, T, D)
values = torch.stack([layer.values for layer in out.past_key_values.layers])
```

The same two helpers (`cache_to_tensors` / `tensors_to_cache` in `stages/kv_utils.py`) are used in Stage 6 to inject a mapped cache back into the model, so extraction and injection are guaranteed to use the same layout.

### 3.3 Calibration Data — No Padding

The WikiText-2 train split (2.5M tokens) is tokenized once and cut into 64 non-overlapping windows of exactly 256 tokens, spaced evenly across the corpus. Every token is real text, so there are no pad tokens to dilute the regression. That gives 64 × 256 = **16,384 token observations** per (layer, head), against 512 input features for the k=4 mapper.

The paper uses 500 × 1024 FineWeb-Edu tokens; its Appendix C shows Wikipedia calibration lands within ~1pp of FineWeb-Edu, and its sample-count sweep is flat down to ~50 sequences.

### 3.4 Output

```
outputs/kv_source.pt   keys/values (24, 64, 2, 256, 64)    403 MB   + input_ids (64, 256)
outputs/kv_target.pt   keys/values (28, 64, 2, 256, 128)   940 MB
```

Run time on the Ryzen 7 7735U: ~6 min for the 0.5B model and ~21 min for the 1.5B model. Peak memory 5.6 GB because one model is loaded at a time.

---

## 4. Stage 3 — RoPE Stripping

**Script:** `stages/02_rope_strip.py`

### 4.1 What is RoPE?

**Rotary Position Embedding** (Su et al., 2024) is how modern transformers encode token position. Instead of adding a fixed positional vector to the input, RoPE rotates the Query and Key vectors by an angle proportional to the token's position.

The rotation for position `t` and frequency dimension `i` uses angle:

```
θ_i = 1 / (rope_base ^ (2i / D))    for i = 0, 1, ..., D/2 - 1
```

For Qwen2.5, `rope_base = 1,000,000`. The very large base means the angles change slowly with position — enabling very long contexts.

The full angle at position `t` for dimension pair `i` is `t · θ_i`.

### 4.2 Forward RoPE (what the model applies)

Qwen2.5 uses the "split-half" rotation convention:

```
Given key vector k of dimension D, split into two halves:
    k1 = k[:D/2]    (first half)
    k2 = k[D/2:]    (second half)

rotate_half(k) = concat(-k2, k1)

k_rotated = k * cos(t·θ) + rotate_half(k) * sin(t·θ)
```

Where `cos(t·θ)` and `sin(t·θ)` are vectors of length D (each angle repeated twice — once per half).

**Concrete example** at position t=3, D=4 (simplified):

```
k = [a, b, c, d]
rotate_half(k) = [-c, -d, a, b]
cos = [cos(3θ_0), cos(3θ_0), cos(3θ_1), cos(3θ_1)]
sin = [sin(3θ_0), sin(3θ_0), sin(3θ_1), sin(3θ_1)]

k_rotated = [a·cos - c·sin,
             b·cos - d·sin,
             c·cos + a·sin,
             d·cos + b·sin]
```

This is mathematically equivalent to a 2D rotation in each (k[i], k[i+D/2]) plane.

### 4.3 Why Strip RoPE Before Mapping?

If we train the mapper on rotated keys, the mapper learns:
> "source key at position 5 → target key at position 5"

This is **position-dependent** — it won't generalise to different prompt lengths or positions.

By stripping RoPE first, the mapper learns:
> "source content representation → target content representation"

This transfers across any context length, exactly as the paper claims.

### 4.4 Inverse RoPE (what we apply)

Since R_θ(t) is an **orthogonal matrix** (rotation matrices always are), its inverse equals its transpose. Transposing a rotation matrix just negates the angle: `R(-θ) = R(θ)^T`.

In terms of the formula, negating the angle flips the sign of sin:

```
k_content = k_rotated * cos(t·θ) - rotate_half(k_rotated) * sin(t·θ)
```

This is the forward formula with `+sin` changed to `-sin`. That's the entire implementation.

### 4.5 Sanity Checks

The script verifies three things:

| Check | Result | What it proves |
|---|---|---|
| Shape unchanged | ✅ `(24,64,2,256,64)` → `(24,64,2,256,64)` | No accidental reshaping |
| Mean \|Δ\| > 0 | ✅ 0.789 (source), 0.542 (target) | Keys changed, so a rotation was removed |
| Round-trip error < 1e-4 | ✅ 3.05e-05 | Inverse is mathematically exact; only floating-point noise |

**Round-trip check:** strip RoPE → re-apply RoPE → compare to original. Max absolute error of 3.05e-05 means no information was lost — the transformation is reversible.

---

## 5. Stage 4 — Top-k Source Layer Selection

**Script:** `stages/03_layer_select.py`

### 5.1 The Problem

The source model has 24 layers. The target has 28. We cannot simply pair them 1:1. We need to discover which source layers are most *linearly predictive* of each target layer.

We use **R² (coefficient of determination)** from linear regression as the predictability score.

### 5.2 Ordinary Least Squares (OLS) Regression

For each pair (source layer `l_s`, target layer `l_t`, head `h`):

1. Flatten calibration data into a matrix:

```
X: (N×T, D_src) = (64×256, 64) = (16384, 64)    source keys (or values)
Y: (N×T, D_tgt) = (64×256, 128) = (16384, 128)  target keys (or values)
```

2. Centre both matrices (subtract column means):

```
X_c = X - mean(X, axis=0)
Y_c = Y - mean(Y, axis=0)
```

3. Solve the **ridge regression** (OLS + small regularisation λ=0.01):

```
W* = (X_c^T · X_c + λI)^{-1} · X_c^T · Y_c
```

The `λI` term (Tikhonov regularisation) prevents numerical instability when `X_c^T · X_c` is near-singular. With λ=0.01 this has negligible impact on the fit.

4. Compute predictions:

```
Y_hat = X_c · W*
```

### 5.3 R² Score

**R²** measures what fraction of the variance in Y is explained by the linear model:

```
SS_res = sum((Y_c - Y_hat)^2)    ← residual sum of squares (unexplained variance)
SS_tot = sum(Y_c^2)              ← total sum of squares (total variance)

R² = 1 - SS_res / SS_tot
```

- R² = 1.0 → perfect linear prediction (source perfectly encodes target)
- R² = 0.0 → no linear relationship (source tells us nothing about target)
- R² = 0.5 → source explains 50% of target's variance

We compute R² per output dimension (each of the 128 target dimensions) then average.

### 5.4 Top-k Selection

We run the regression for keys and for values: 2 × 24 × 28 × 2 = 2688 regressions. Averaging across the 2 heads gives one key matrix and one value matrix of shape `(28, 24)`. Following paper §3.2, the selection score is their mean.

For each of the 28 target layers, we pick the 4 source layers with the highest score (k=4, following the paper's observation that R² climbs steeply from k=1 to k=4):

```python
top_k_vals, top_k_idx = torch.topk(r2_matrix, k=4, dim=1)
# top_k_idx shape: (28, 4)
```

### 5.5 Results

```
Mean key R²   : 0.401   (all 672 layer pairs, single source layer, single head)
Mean value R² : 0.140
Best cell     : 0.472   (target layer 1 ← source layers 0–3)
Most-selected source layer: 15 (in the top-4 of 14 of the 28 target layers)
```

**Layer alignment pattern (from the heatmap):**

| Target layer range | Preferred source layers | Interpretation |
|---|---|---|
| 0–2 (early) | 0–3 | Both models build similar low-level token representations |
| 3–9 | 6–8 | Early-middle layers of the 0.5B feed a wide band of the 1.5B |
| 10–21 (middle) | 11–15 | Semantic core of the 0.5B maps to the middle of the 1.5B |
| 22–27 (late) | 20–23 | Late layers align with late layers |

This "soft diagonal" is the cross-model linear structure the paper describes (its Figure 2).

### 5.6 How This Compares to the Paper

The paper's single-layer key R² is 0.56 on Qwen3 14B→32B and lower on more distant pairs. Ours is 0.40 for keys and 0.14 for values. Two reasons it is lower:
1. This is a **non-matched-KV** pair (head_dim 64 → 128). A single 64-dim source head must predict a 128-dim target head, which the paper never tests.
2. The 0.5B → 1.5B pair has a 3× parameter gap and a different width, so the representations are less aligned than 14B → 32B.

Stage 5 shows that concatenating 4 source layers and both heads lifts R² substantially (keys 0.63, values 0.41 held-out).

---

## 6. Results Summary

| Stage | Script | Input | Output | Key Metric |
|---|---|---|---|---|
| 1 & 2 | `01_extract_kv.py` | WikiText-2 train | `kv_source.pt`, `kv_target.pt` | Shape `(L, 64, 2, 256, D)` |
| 3 | `02_rope_strip.py` | `kv_*.pt` | `kv_*_norope.pt` | Round-trip error 3.05e-05 |
| 4 | `03_layer_select.py` | `kv_*_norope.pt` | `r2_matrix.pt`, `r2_kv.pt`, `top_k_mapping.pt` | Single-layer R²: K 0.40, V 0.14 |
| 5 | `04_fit_mapper.py` | `kv_*_norope.pt`, `r2_matrix.pt` | `mapper_k4.pt` | Held-out R²: K 0.63, V 0.41 |
| 6 | `05_benchmark.py` | `mapper_k4.pt`, WikiText-2 test | `benchmark_k4.json` | PPL 13.78 vs 9.60 standalone, 19.50 no prefix |

**Deviations from the paper (to be aware of):**

| Paper | This Replication | Impact |
|---|---|---|
| 500 sequences × 1024 tokens | 64 sequences × 256 tokens | Paper's N sweep is flat down to ~50 sequences |
| FineWeb-Edu dataset | WikiText-2 (Wikipedia) | Paper App. C: Wikipedia within ~1pp |
| Matched-KV (head_dim equal) | Non-matched (64 → 128) | Untested in the paper; W* is non-square |
| k swept per pair (k = 8–all) | k = 4 fixed | Likely leaves accuracy on the table (see §9) |
| ARC/HellaSwag/MMLU/GSM8K accuracy | Prefix-conditioned perplexity only | No downstream accuracy numbers yet |
| H100 GPUs, bf16 | Laptop CPU, bf16 | Latency ratios are not comparable |

---

## 7. Stage 5 — Ridge Mapper

**Script:** `stages/04_fit_mapper.py`

### 7.1 The Model

For target layer `l`, the input is every head of its top-k source layers, concatenated (paper Eq. 5):

```
X_K^l = [ K_s^{l1} ‖ K_s^{l2} ‖ K_s^{l3} ‖ K_s^{l4} ]   each block = 2 heads × 64 dims
      → one row per token, 4 × 2 × 64 = 512 features
```

Each target head `h` gets its own regression (paper Eq. 3–4):

```
K̂_t^{l,h} = X_K^l · W_K^{l,h} + b_K^{l,h}        W_K^{l,h} : 512 × 128
W* = (Xc^T Xc + λI)^{-1} Xc^T Yc,   b = Ȳ − X̄ W*,   λ = 0.01
```

Values are mapped the same way with their own weights. All heads of a layer share `X`, so solving them together as one multi-output system gives exactly the per-head solutions — the columns of a ridge solve are independent. `X^T X` is formed in float64 because the 4 selected layers are strongly correlated and the matrix is close to singular.

Keys are fit in **content space**: source keys have source RoPE removed, target keys have target RoPE removed. Stage 6 re-applies the target's RoPE at the right positions.

### 7.2 Size

28 layers × 2 (K,V) × 512 × 256 weights + bias = **7.35M parameters, 29 MB**. The mapper is tiny next to the 1.5B model.

### 7.3 Results (k = 4, 56 chunks fit, 8 chunks held out)

| | Train R² | Held-out R² |
|---|---|---|
| Keys | 0.707 | **0.634** |
| Values | 0.519 | **0.408** |

Key R² is flat across depth (0.57–0.74 held-out). Value R² is best in the first layers (0.61–0.66 for target layers 0–1) and weakest in the middle and late layers (0.25 at target layer 15). The 0.07–0.11 drop from train to held-out is mild overfitting from 512 features per layer. The saved mapper is refit on all 64 chunks.

---

## 8. Stage 6 — Benchmark

**Script:** `stages/05_benchmark.py`. Evaluation uses the WikiText-2 **test** split, which the mapper never saw.

### 8.1 Protocol — Prefix-Conditioned Perplexity

This is the paper's Appendix D protocol, scaled to CPU (paper: 1024 + 1024 tokens; here 256 + 128, 16 chunks, 2,032 scored tokens). Each chunk is split into a **prefix** and a **continuation**. The 1.5B model scores the identical continuation tokens five ways; only the prefix KV cache differs:

```
chunk:  [ ───────── prefix: 256 tokens ───────── | ── continuation: 128 tokens ── ]
                         │                                    │
           whose KV cache is used?                 1.5B scores these tokens
```

| Mode | What fills the prefix cache |
|---|---|
| target standalone | 1.5B prefills the prefix itself — the ceiling |
| GT cache round-trip | 1.5B's own cache → strip RoPE → re-apply RoPE → rebuilt cache. Must equal the ceiling, otherwise the injection code is wrong |
| **transfer** | 0.5B prefills; strip source RoPE → mapper → apply target RoPE → cache |
| target, no prefix | Nothing — the floor |
| source standalone | 0.5B continues on its own (reference) |

**Prefix benefit retained** = (PPL_no_prefix − PPL_mode) / (PPL_no_prefix − PPL_standalone). It is 100% for a perfect cache and 0% for a cache worth nothing — a perplexity analogue of the paper's floor-normalized retention.

### 8.2 Results (k = 4)

| Mode | PPL | Benefit kept | Top-1 agreement with standalone | KL from standalone |
|---|---|---|---|---|
| target standalone | 9.60 | 100% | 100% | 0 |
| GT cache round-trip | 9.60 | 100% | 100% | 0 |
| **transfer** | **13.78** | **57.8%** | **77.3%** | **0.379** |
| target, no prefix | 19.50 | 0% | 71.2% | 0.699 |
| source standalone | 13.31 | 62.5% | 70.7% | 0.463 |

Held-out KV R² on the test prefixes: keys **0.66**, values **0.45**, close to the Stage 5 validation numbers, so the mapper generalizes from train to test text.

**Interpretation:**

1. **The plumbing is exact.** The round-trip row matches standalone to two decimals, so every gap in the transfer row comes from the mapper, not from RoPE or cache handling.
2. **The transferred cache carries real information.** Perplexity falls from 19.5 to 13.8, recovering 58% of what the prefix is worth.
3. **It is not yet better than not switching.** The 0.5B model continuing alone scores 13.31, slightly better than the 1.5B model with a mapped cache. The mapped cache does make the 1.5B behave more like itself than the 0.5B does (top-1 agreement 77% vs 71%, KL 0.38 vs 0.46), but on perplexity the transfer does not yet pay off.
4. **Values are the bottleneck.** Value R² is ~0.2 lower than key R², the same K > V gap the paper reports (its §2.3, pattern 4).

### 8.3 Latency

Median of 3 runs, 8 threads, bf16:

| Tokens | 1.5B re-prefill | Mapper | Ratio | 0.5B prefill (for reference) |
|---|---|---|---|---|
| 64 | 4,366 ms | 14.9 ms | 294× | 1,382 ms |
| 256 | 16,530 ms | 45.4 ms | 364× | 4,958 ms |
| 1024 | 65,992 ms | 166.2 ms | 397× | 18,760 ms |

As in the paper (§4.7), this compares re-prefill with applying the mapper to a cache the source **already has**. The ratio is much larger than the paper's 4–25× because this CPU has no native bf16 matmul, which slows the 1.5B forward pass far more than the mapper's small float32 matmuls. Read it as "the mapper is negligible next to prefill", not as a hardware-comparable speedup.

### 8.4 Walkthrough

`python stages/walkthrough.py` follows one prompt through every step. On the default prompt *"…Paris is the capital of France, and Rome is the capital of"*, the next-token predictions are:

| Cache | Top predictions |
|---|---|
| 1.5B, own prefill | ' Italy' 97% |
| 1.5B, transferred KV | ' Italy' 82%, ' the' 10%, ' France' 2% |
| 1.5B, no prefix | ' the' 32% |

Mean cosine between mapped and real KV for that prompt: keys 0.93, values 0.69.

---

## 9. What Comes Next (per the paper)

The paper continues past benchmarking in this order:

1. **Sweep k** over {1, 2, 4, 6, 8, 10, 12, 16, 20, 24, all} and pick k by downstream accuracy (§4.1). Here: `python stages/04_fit_mapper.py --k 8` then `python stages/05_benchmark.py --k 8`.
2. **Downstream accuracy and retention** on ARC-Challenge, HellaSwag, WinoGrande, MMLU (5-shot) and GSM8K (8-shot CoT) via lm-evaluation-harness defaults (§4.2, Table 1).
3. **Component ablation** — remove inference-time RoPE handling, cross-layer selection (k=1), and ridge in turn (§4.3, Table 2).
4. **Nonlinear MLP mapper** — two 1024-unit ReLU layers per (layer, head, K|V), Adam, 20 epochs; recovers up to +37pp HellaSwag on pairs where ridge fails (§4.4).
5. **Diagnostics** — attention-output cosine (predicts retention better than R²) and error concentration in attention-sensitive subspaces (§4.5).
6. **Multi-turn handoff** on CoQA, alternating models across turns, and the large-to-small direction (§4.6).

---

## 10. Sources

| Source | Used For |
|---|---|
| Heo et al. (2026). *Cross-Model KV Cache Transfer in LLM Families.* NVIDIA. | Primary reference — full paper in `paper/` |
| Su et al. (2024). *RoFormer: Enhanced Transformer with Rotary Position Embedding.* | RoPE math in Stage 3 |
| Qwen2.5 technical report. Alibaba Cloud, 2024. | Source/target model architecture |
| Hugging Face Transformers library. | Model loading, tokenization (`AutoModelForCausalLM`) |
| PyTorch documentation. `torch.linalg.solve`. | Ridge regression solver in Stages 4 & 5 |
| Tikhonov, A. N. (1963). *Solution of incorrectly formulated problems.* | Ridge regularisation theory |
