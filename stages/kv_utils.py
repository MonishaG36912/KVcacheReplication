"""
Shared helpers for the pipeline stages.

Everything that more than one stage needs lives here:
    * model identifiers
    * the WikiText-2 text corpus (calibration = train split, evaluation = test split)
    * reading / writing a Hugging Face DynamicCache as plain tensors
    * RoPE strip / re-apply at arbitrary positions
    * the feature layout and application of the Stage 5 ridge mapper

Run the stages from the repo root (python stages/0X_*.py). Python puts the
script's own folder on sys.path, so `import kv_utils` works from any stage.
"""

import torch
from transformers import DynamicCache

SOURCE_MODEL_ID = "Qwen/Qwen2.5-0.5B"
TARGET_MODEL_ID = "Qwen/Qwen2.5-1.5B"

CORPUS_REPO = "Salesforce/wikitext"
CORPUS_FILES = {
    "train": "wikitext-2-raw-v1/train-00000-of-00001.parquet",
    "validation": "wikitext-2-raw-v1/validation-00000-of-00001.parquet",
    "test": "wikitext-2-raw-v1/test-00000-of-00001.parquet",
}


# ── corpus ────────────────────────────────────────────────────────────────────

def load_corpus_ids(tokenizer, split: str) -> torch.Tensor:
    """
    Download one WikiText-2 split (a few MB, cached by huggingface_hub) and
    return the whole split tokenized as a single 1-D LongTensor.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(CORPUS_REPO, CORPUS_FILES[split], repo_type="dataset")
    lines = pq.read_table(path).column("text").to_pylist()
    text = "".join(lines)  # every WikiText line already ends with "\n"
    return tokenizer(text, return_tensors="pt").input_ids[0]


def make_chunks(ids: torch.Tensor, n_chunks: int, chunk_len: int) -> torch.Tensor:
    """
    Cut `n_chunks` non-overlapping windows of exactly `chunk_len` tokens,
    spread evenly across the corpus so the chunks cover many articles.
    No padding is ever needed, so every token is a real token.
    """
    stride = ids.numel() // n_chunks
    if stride < chunk_len:
        raise ValueError(f"Corpus has {ids.numel()} tokens, need {n_chunks} x {chunk_len}.")
    return torch.stack([ids[i * stride: i * stride + chunk_len] for i in range(n_chunks)])


# ── DynamicCache <-> tensors ──────────────────────────────────────────────────

def cache_to_tensors(cache) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Stack a DynamicCache into two tensors of shape (L, B, H, T, D).
    These are exactly what the model stores: keys are RoPE-rotated.
    """
    keys = torch.stack([layer.keys for layer in cache.layers])
    values = torch.stack([layer.values for layer in cache.layers])
    return keys, values


def tensors_to_cache(keys: torch.Tensor, values: torch.Tensor, config) -> DynamicCache:
    """Build a DynamicCache from (L, B, H, T, D) tensors (inverse of cache_to_tensors)."""
    cache = DynamicCache(config=config)
    for l in range(keys.shape[0]):
        cache.update(keys[l].contiguous(), values[l].contiguous(), l)
    return cache


# ── RoPE ──────────────────────────────────────────────────────────────────────

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Qwen2 / LLaMA rotate_half: [x1, x2] -> [-x2, x1] on the last dimension."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope_tables(positions: torch.Tensor, head_dim: int, rope_theta: float):
    """
    cos / sin tables of shape (T, head_dim) for the given absolute positions,
    computed in float32 exactly like Qwen2RotaryEmbedding (default rope type).
    """
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(positions.float(), inv_freq)        # (T, D/2)
    emb = torch.cat([angles, angles], dim=-1)                # (T, D)
    return emb.cos(), emb.sin()


def strip_rope(keys: torch.Tensor, rope_theta: float, start_pos: int = 0) -> torch.Tensor:
    """
    k_content = R(t)^-1 k_rot for keys of shape (..., T, D) at positions
    start_pos .. start_pos+T-1. R is orthogonal, so the inverse is the
    rotation by the negative angle.
    """
    T, D = keys.shape[-2:]
    cos, sin = rope_tables(torch.arange(start_pos, start_pos + T), D, rope_theta)
    return keys * cos - rotate_half(keys) * sin


def apply_rope(keys: torch.Tensor, rope_theta: float, start_pos: int = 0) -> torch.Tensor:
    """k_rot = R(t) k_content, the forward rotation the model applies."""
    T, D = keys.shape[-2:]
    cos, sin = rope_tables(torch.arange(start_pos, start_pos + T), D, rope_theta)
    return keys * cos + rotate_half(keys) * sin


def rope_theta_of(config) -> float:
    params = getattr(config, "rope_parameters", None) or {}
    if params.get("rope_type", "default") != "default":
        raise NotImplementedError(f"Only default RoPE is supported, got {params}")
    return float(params.get("rope_theta", getattr(config, "rope_theta", 10000.0)))


# ── mapper feature layout ─────────────────────────────────────────────────────

def source_features(kv: torch.Tensor, layer_idx: torch.Tensor) -> torch.Tensor:
    """
    Paper Eq. 5: X = [K_s^{l1} || ... || K_s^{lk}] where each block is the
    concatenation of ALL source KV heads of that layer.

    kv        : (L_src, N, H, T, D)
    layer_idx : (k,) selected source layers for one target layer
    returns   : (N*T, k*H*D)  one row per token
    """
    sel = kv[layer_idx]                               # (k, N, H, T, D)
    k, N, H, T, D = sel.shape
    return sel.permute(1, 3, 0, 2, 4).reshape(N * T, k * H * D)


def target_rows(kv_layer: torch.Tensor) -> torch.Tensor:
    """(N, H, T, D) -> (N*T, H*D). Column block h*D:(h+1)*D is target head h."""
    N, H, T, D = kv_layer.shape
    return kv_layer.permute(0, 2, 1, 3).reshape(N * T, H * D)


def rows_to_heads(rows: torch.Tensor, N: int, T: int, H: int) -> torch.Tensor:
    """Inverse of target_rows: (N*T, H*D) -> (N, H, T, D)."""
    return rows.reshape(N, T, H, -1).permute(0, 2, 1, 3)


def apply_mapper(mapper: dict, src_keys_c: torch.Tensor, src_vals: torch.Tensor):
    """
    Map source content-space KV to target content-space KV.

    src_keys_c, src_vals : (L_src, N, H_src, T, D_src), keys RoPE-stripped
    returns              : two tensors (L_tgt, N, H_tgt, T, D_tgt), float32
    """
    N, T = src_keys_c.shape[1], src_keys_c.shape[3]
    H_t = mapper["tgt_kv_heads"]
    src_keys_c = src_keys_c.float()
    src_vals = src_vals.float()
    out_k, out_v = [], []
    for l, idx in enumerate(mapper["sources"]):
        Xk = source_features(src_keys_c, idx)
        Xv = source_features(src_vals, idx)
        yk = torch.addmm(mapper["b_K"][l], Xk, mapper["W_K"][l])
        yv = torch.addmm(mapper["b_V"][l], Xv, mapper["W_V"][l])
        out_k.append(rows_to_heads(yk, N, T, H_t))
        out_v.append(rows_to_heads(yv, N, T, H_t))
    return torch.stack(out_k), torch.stack(out_v)


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Mean over output columns of 1 - SS_res/SS_tot, NOT clipped, so a
    mapper that is worse than predicting the mean shows up as negative
    (the paper reports negative R² on its failure pairs).
    """
    y_true = y_true.double()
    resid = ((y_true - y_pred.double()) ** 2).sum(0)
    total = ((y_true - y_true.mean(0)) ** 2).sum(0)
    return (1.0 - resid / (total + 1e-12)).mean().item()
