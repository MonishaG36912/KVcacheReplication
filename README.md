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
│   └── 05_benchmark.py              # Stage 6: Perplexity, KV R² and latency in the real target model
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

## Results So Far

| Stage | Status | Key Result |
|---|---|---|
| KV Extraction | ✅ Complete | Tensors shape `(L, 15, 2, 256, D)` saved |
| RoPE Stripping | ✅ Complete | Round-trip error 3.05e-05, well below 1e-4 |
| Layer Selection | ✅ Complete | Mean R² = **0.777** across all 672 layer pairs |
| Ridge Mapper | 🔲 Pending | Stage 5 — `04_fit_mapper.py` |
| Benchmarking | 🔲 Pending | Stage 6 — `05_benchmark.py` |

**Mean R² of 0.777** means the source model's KV cache can linearly predict ~78% of the variance in the target model's KV cache — strong evidence the transfer will work.

---

## References

- Heo et al. (2026). *Cross-Model KV Cache Transfer in LLM Families.* NVIDIA. [`paper/`](paper/)
- Qwen2.5 model family: [huggingface.co/Qwen](https://huggingface.co/Qwen)
- Su et al. (2024). *RoFormer: Enhanced Transformer with Rotary Position Embedding.*
- Hugging Face Transformers: [github.com/huggingface/transformers](https://github.com/huggingface/transformers)
