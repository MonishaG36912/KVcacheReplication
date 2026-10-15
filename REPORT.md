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
7. [What Remains](#7-what-remains)
8. [Sources](#8-sources)

---

## 1. Project Goal

When two language models in the same family (one small, one large) process the same text, they produce internal Key-Value (KV) tensors at every attention layer. The paper asks: **can we train a linear function that converts the small model's KV tensors into the large model's expected KV format**, so the large model never has to re-read the prompt?

If yes, this skips the **prefill** step entirely when switching models — the most expensive part of serving long prompts.

The pipeline has six stages:

```
Load Models → Extract KV → Strip RoPE → Select Layers → Fit Mapper → Benchmark
   Stage 1        Stage 2     Stage 3       Stage 4       Stage 5     Stage 6
```

Stages 1–4 are complete. This report covers them in full.

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
| Qwen2.5-0.5B (source) | 24 | 2 | 64 | (24, 15, 2, 256, 64) |
| Qwen2.5-1.5B (target) | 28 | 2 | 128 | (28, 15, 2, 256, 128) |

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

### 3.2 The Hook Mechanism

PyTorch modules have a `register_forward_hook` method that fires a callback every time that module runs. We use this to intercept K and V without modifying model code.

**The problem:** Qwen2Attention computes K and V internally as local variables — they are not stored anywhere accessible after the forward pass.

**The solution — two-level interception:**

```
Layer forward pass
    │
    └── k_proj.forward(X)   ← inner hook fires here, captures k_raw
    └── v_proj.forward(X)   ← inner hook fires here, captures v_raw
    │
    (reshape k_raw → key_states, stored as module attribute)
    │
    └── self_attn.forward() ← outer hook fires here, reads key_states
                               appends to accumulator list
```

**Step 1 — Project and capture raw output:**

`k_proj` produces shape `(B, T, n_kv_heads × D)` — a flat vector per token.

**Step 2 — Reshape into head-separated form:**

```python
k_raw = captured["k_raw"]               # shape: (B, T, n_kv_heads × D)
B, T, _ = k_raw.shape
key_states = k_raw.view(B, T, n_kv_heads, D).transpose(1, 2)
# result shape: (B, n_kv_heads, T, D)
```

`transpose(1, 2)` swaps the T and H axes so each head's keys are contiguous.

**Step 3 — Accumulate across samples:**

After running all 15 prompts:

```python
# per layer: list of 15 tensors each shape (1, H, T, D)
keys.append(torch.stack([t.squeeze(0) for t in l_keys], dim=0))
# result: (15, H, T, D) per layer
```

Then stack across layers: `(L, 15, H, T, D)`.

### 3.3 Padding

Different sentences tokenize to different lengths. To stack them:

```python
tokenizer(text, padding="max_length", max_length=256, truncation=True)
```

Every sentence is padded with `[PAD]` tokens to exactly 256. The `attention_mask` marks padding positions as 0 so the model ignores them during attention computation.

### 3.4 Example

Input: `"The theory of general relativity..."` (36 tokens before padding → padded to 256)

After passing through the source model:

```
layer 0, head 0, position 0:  k = [0.021, -0.043, 0.018, ..., 0.067]  (64 values)
layer 0, head 0, position 1:  k = [-0.012, 0.089, -0.034, ..., 0.011]
...
layer 23, head 1, position 255: k = [...]  (padding position, near-zero after masking)
```

Saved to `outputs/kv_source.pt`:
```
{
  "keys":   tensor of shape (24, 15, 2, 256, 64),  47 MB
  "values": tensor of shape (24, 15, 2, 256, 64),  47 MB
}
```

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
| Shape unchanged | ✅ `(24,15,2,256,64)` → `(24,15,2,256,64)` | No accidental reshaping |
| Mean \|Δ\| > 0 | ✅ 0.682 (source), 0.427 (target) | RoPE was actually present and removed |
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
X: (N×T, D_src) = (15×256, 64) = (3840, 64)    source keys
Y: (N×T, D_tgt) = (15×256, 128) = (3840, 128)  target keys
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

After computing R² for all 24 × 28 × 2 = 1344 regressions, we average across the 2 heads to get a single score per `(l_s, l_t)` pair — a matrix of shape `(28, 24)`.

For each of the 28 target layers, we pick the 4 source layers with the highest R² (k=4, following the paper's recommendation that R² plateaus around k=4–6):

```python
top_k_vals, top_k_idx = torch.topk(r2_matrix, k=4, dim=1)
# top_k_idx shape: (28, 4)
```

### 5.5 Results

```
Overall mean R²: 0.7768
Overall max R²:  0.8538  (target layer 15, best source layer 14)
Overall min R²:  0.6573

Most-selected source layers: 14, 15, 18 (appear in top-4 for most target layers)
```

**Layer alignment pattern (from the heatmap):**

| Target layer range | Preferred source layers | Interpretation |
|---|---|---|
| 0–1 (early) | 0, 1, 2, 3 | Both models learn similar low-level token representations |
| 7–18 (middle) | 10–15 | Semantic core of 0.5B maps to wide middle band of 1.5B |
| 23–27 (late) | 20–23 | Both models converge to similar output-space representations |

This "soft diagonal" pattern is exactly what the paper describes as evidence of cross-model linear structure.

### 5.6 Why R² of 0.77 is Good

The paper reports R²=0.56 for its main pair (Qwen3 14B→32B) at k=1, rising to R²=0.79 at k=8.

Our result of **0.777 at k=1** (single best source layer per target) is high because:
1. Qwen2.5-0.5B and 1.5B are architecturally very close — same family, same GQA ratio
2. We measure on calibration data (in-sample) — out-of-sample R² will be somewhat lower

---

## 6. Results Summary

| Stage | Script | Input | Output | Key Metric |
|---|---|---|---|---|
| 1 & 2 | `01_extract_kv.py` | Raw text prompts | `kv_source.pt`, `kv_target.pt` | Shape `(L, 15, 2, 256, D)` |
| 3 | `02_rope_strip.py` | `kv_*.pt` | `kv_*_norope.pt` | Round-trip error 3.05e-05 |
| 4 | `03_layer_select.py` | `kv_*_norope.pt` | `r2_matrix.pt`, `top_k_mapping.pt` | Mean R² = 0.777 |

**Deviations from the paper (to be aware of):**

| Paper | This Replication | Impact |
|---|---|---|
| 500 sequences × 1024 tokens | 15 sequences × 256 tokens | Less robust mapper |
| FineWeb-Edu dataset | Hard-coded sentences | May bias toward academic text |
| Matched-KV (head_dim equal) | Non-matched (64 → 128) | W* is non-square |
| R² averaged over K and V | R² on K only | Minor difference |

---

## 7. What Remains

**Stage 5 — Ridge Mapper (`04_fit_mapper.py`):**
For each target (layer, head) pair, solve:

```
W*_K = (X_K^T · X_K + λI)^{-1} · X_K^T · Y_K
W*_V = (X_V^T · X_V + λI)^{-1} · X_V^T · Y_V
```

Where `X_K` concatenates the top-4 source layers' keys: shape `(N×T, 4 × 2 × 64) = (3840, 512)`.

Then re-apply target RoPE to the mapped keys before storing.

**Stage 6 — Benchmarking (`05_benchmark.py`):**
- Pass the mapped KV as `past_key_values` into the 1.5B model
- Measure Time-to-First-Token (TTFT) vs standalone prefill
- Measure perplexity on held-out text

---

## 8. Sources

| Source | Used For |
|---|---|
| Heo et al. (2026). *Cross-Model KV Cache Transfer in LLM Families.* NVIDIA. | Primary reference — full paper in `paper/` |
| Su et al. (2024). *RoFormer: Enhanced Transformer with Rotary Position Embedding.* | RoPE math in Stage 3 |
| Qwen2.5 technical report. Alibaba Cloud, 2024. | Source/target model architecture |
| Hugging Face Transformers library. | Model loading, tokenization (`AutoModelForCausalLM`) |
| PyTorch documentation. `torch.linalg.solve`. | Ridge regression solver in Stages 4 & 5 |
| Tikhonov, A. N. (1963). *Solution of incorrectly formulated problems.* | Ridge regularisation theory |
