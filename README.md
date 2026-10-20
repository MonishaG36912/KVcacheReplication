# KVcacheSWE — Cross-Model KV Cache Transfer

A local CPU replication of the NVIDIA paper:
> **"Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse"**
> Heo et al., NVIDIA, 2026 — [`paper/nvidia_cross_model_kv_2026.pdf`](paper/nvidia_cross_model_kv_2026.pdf)

---

## What This Project Does

Large language models spend significant time on **prefill** — the forward pass that reads the entire prompt before generating the first token. When a serving system switches between two models in the same family (e.g. a small model for drafting, a large model for final output), the large model must re-read the full prompt from scratch even though the small model already processed it.

This project replicates the paper's core idea: **train a small linear mapper that converts the small model's KV cache into the large model's expected format**, skipping re-prefill entirely.

---

## Hardware & Environment

| Component | Value |
|---|---|
| CPU | AMD Ryzen 7 7735U (8 cores) |
| RAM | 16 GB system RAM |
| GPU | None — pure CPU execution |
| OS | Ubuntu 22.04 via WSL2 on Windows |
| Python | 3.10 |
| PyTorch | CPU-only, bfloat16 weights |

---

## Models

| Role | Model | RAM Usage | Layers | KV Heads | Head Dim |
|---|---|---|---|---|---|
| Source | `Qwen/Qwen2.5-0.5B` | ~1 GB | 24 | 2 | 64 |
| Target | `Qwen/Qwen2.5-1.5B` | ~3 GB | 28 | 2 | 128 |

Both models are from the same Qwen2.5 family and share a tokenizer, making them valid transfer candidates. Note that head_dim differs (64 vs 128), so this is a **non-matched-KV** pair by the paper's strict definition — the mapper weight matrix is non-square.

---

## Project Structure

```
KVcacheSWE/
├── README.md                        # This file
├── REPORT.md                        # Full mathematical report on each stage
├── requirements.txt                 # Python dependencies
├── .gitignore
│
├── paper/
│   └── nvidia_cross_model_kv_2026.pdf   # Reference paper
│
├── stages/
│   ├── kv_utils.py                  # Shared helpers: corpus, cache <-> tensors, RoPE, mapper layout
│   ├── 01_extract_kv.py             # Stage 1 & 2: Load models, extract KV caches on WikiText-2 train
│   ├── 02_rope_strip.py             # Stage 3: Remove RoPE from Key vectors
│   ├── 03_layer_select.py           # Stage 4: Top-k source layer selection via K+V R²
│   ├── 04_fit_mapper.py             # Stage 5: Closed-form per-head ridge mapper
│   ├── 05_benchmark.py              # Stage 6: Perplexity, KV R² and latency in the real target model
│   └── walkthrough.py               # Teaching demo: one prompt through every step
│
└── outputs/
    ├── .gitkeep                     # Keeps folder tracked; .pt files are gitignored
    ├── inspect_outputs.py           # Utility to inspect saved .pt tensor files
    ├── kv_source.pt                 # [generated] KV cache tensors (RoPE applied), source model
    ├── kv_target.pt                 # [generated] KV cache tensors (RoPE applied), target model
    ├── kv_source_norope.pt          # [generated] RoPE-stripped KV, source model
    ├── kv_target_norope.pt          # [generated] RoPE-stripped KV, target model
    ├── r2_matrix.pt                 # [generated] Selection score (L_tgt × L_src), mean of K and V R²
    ├── r2_kv.pt                     # [generated] Separate K and V R² matrices
    ├── top_k_mapping.pt             # [generated] Top-4 source layers per target layer
    ├── mapper_k{k}.pt               # [generated] Ridge mapper weights for a given k
    └── benchmark_k{k}.json          # [generated] Stage 6 results (tracked in git)
```

---

## Setup

```bash
# Clone and enter the repo
git clone https://github.com/MonishaG36912/KVcacheSWE.git
cd KVcacheSWE

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> **First run only:** models (~4 GB total) and the WikiText-2 corpus (~10 MB) are downloaded automatically from Hugging Face and cached in `~/.cache/huggingface/hub/`. Subsequent runs use the local cache.

---

## Running the Pipeline

Run each stage in order from the repo root. Each stage saves its output to `outputs/` for the next stage to load.

```bash
# Stage 1 & 2 — Extract KV caches on 64 × 256 WikiText-2 train tokens
python stages/01_extract_kv.py

# Stage 3 — Strip RoPE from Key vectors
python stages/02_rope_strip.py

# Stage 4 — Select top-4 source layers per target layer via K+V R²
python stages/03_layer_select.py

# Stage 5 — Fit the ridge mapper (add --k 8 or --k all to try other k)
python stages/04_fit_mapper.py

# Stage 6 — Benchmark the mapped cache inside Qwen2.5-1.5B (add --k to match Stage 5)
python stages/05_benchmark.py

# Inspect any saved .pt file
python outputs/inspect_outputs.py
```

Each stage runs one model at a time where possible, so peak memory stays under ~6 GB.

---

## Results

| Stage | Status | Key Result |
|---|---|---|
| KV Extraction | ✅ Complete | 64 × 256 WikiText-2 tokens, tensors `(L, 64, 2, 256, D)` |
| RoPE Stripping | ✅ Complete | Round-trip error 3.05e-05, well below 1e-4 |
| Layer Selection | ✅ Complete | Single-layer R²: keys 0.40, values 0.14 |
| Ridge Mapper (k=4) | ✅ Complete | Held-out R²: keys **0.63**, values **0.41** |
| Benchmarking | ✅ Complete | Transfer keeps **58%** of the prefix's perplexity benefit |

### Stage 6 — prefix-conditioned perplexity (WikiText-2 test, 16 × (256 prefix + 128 scored))

| How the 1.5B model gets the prefix | PPL ↓ | Prefix benefit kept | Top-1 agrees with 1.5B |
|---|---|---|---|
| Its own prefill (ceiling) | 9.60 | 100% | 100% |
| Own cache via strip → re-RoPE (plumbing check) | 9.60 | 100% | 100% |
| **Transferred from 0.5B via mapper** | **13.78** | **58%** | **77%** |
| No prefix at all (floor) | 19.50 | 0% | 71% |
| *(reference)* 0.5B continues on its own | 13.31 | 62% | 71% |

**Reading it:** the mapped cache clearly carries the prefix's meaning — perplexity drops from 19.5 to 13.8 — but about 40% of the benefit is lost. The 1.5B model with a mapped cache agrees with its own predictions more often than the 0.5B model does, yet its perplexity is still slightly worse than just letting the 0.5B model continue. Values are the weak spot (R² 0.41 vs 0.63 for keys), matching the paper's finding that V is harder than K.

**Latency (CPU):** applying the mapper takes 15–166 ms vs 4.4–66 s for 1.5B re-prefill at 64–1024 tokens. The ~300–400× ratio is inflated because bf16 matmuls are slow on this CPU; the paper measures 4–25× on H100s.

**One-prompt walkthrough:** `python stages/walkthrough.py` follows a single sentence through every step and prints the next-token predictions from each cache.

---

## References

- Heo et al. (2026). *Cross-Model KV Cache Transfer in LLM Families.* NVIDIA. [`paper/`](paper/)
- Qwen2.5 model family: [huggingface.co/Qwen](https://huggingface.co/Qwen)
- Su et al. (2024). *RoFormer: Enhanced Transformer with Rotary Position Embedding.*
- Hugging Face Transformers: [github.com/huggingface/transformers](https://github.com/huggingface/transformers)
