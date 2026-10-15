"""
Stage 1 & 2 — KV Cache Extraction
Loads Qwen2.5-0.5B (source) and Qwen2.5-1.5B (target), runs calibration
prompts through each model, and extracts Key/Value tensors from every
attention layer via forward hooks.

Saved outputs (in ./outputs/):
    kv_source.pt  — dict { "keys": [L_src, N, H, T, D],
                           "values": [L_src, N, H, T, D] }
    kv_target.pt  — same shape schema for the target model

Tensor dimension legend:
    L = num layers, N = num calibration samples, H = num KV heads,
    T = sequence length, D = head_dim
"""

import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── reproducibility & performance ────────────────────────────────────────────
torch.set_grad_enabled(False)
torch.manual_seed(42)
DEVICE = torch.device("cpu")
DTYPE  = torch.bfloat16      # halves RAM vs float32; CPU supports bfloat16

# ── model identifiers ────────────────────────────────────────────────────────
SOURCE_MODEL_ID = "Qwen/Qwen2.5-0.5B"
TARGET_MODEL_ID = "Qwen/Qwen2.5-1.5B"

# ── calibration prompts (hard-coded so no dataset download is needed) ────────
CALIBRATION_TEXTS = [
    "The theory of general relativity describes gravity as the curvature of spacetime.",
    "Machine learning models learn patterns from large datasets to make predictions.",
    "The Amazon rainforest produces about 20% of the world's oxygen supply.",
    "Quantum computers use qubits to perform calculations exponentially faster.",
    "The French Revolution began in 1789 and fundamentally changed European politics.",
    "Neural networks are composed of layers of interconnected artificial neurons.",
    "Climate change is accelerating the melting of polar ice caps worldwide.",
    "The human genome contains approximately 3 billion base pairs of DNA.",
    "Transformers use self-attention mechanisms to model long-range dependencies.",
    "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose.",
    "The Silk Road was an ancient trade network connecting China to the Mediterranean.",
    "Large language models are pretrained on trillions of tokens of text data.",
    "Plate tectonics explains the movement of Earth's crustal plates over time.",
    "The mitochondria is often called the powerhouse of the cell in biology.",
]

MAX_SEQ_LEN = 256   # truncate / pad each prompt to this many tokens


# ── hook infrastructure ───────────────────────────────────────────────────────

class KVHookManager:
    """
    Attaches one forward hook per attention layer and accumulates the
    key_states / value_states tensors produced during each forward pass.

    Qwen2.5 attention modules expose key_states and value_states as the
    2nd and 3rd positional outputs of their forward() call.  If the
    module returns a tuple we grab indices 1 and 2; if it returns an
    object we fall back to attribute access.
    """

    def __init__(self, model: nn.Module):
        self.keys_per_layer: list[list[torch.Tensor]] = []   # [layer][sample]
        self.vals_per_layer: list[list[torch.Tensor]] = []
        self._hooks: list = []
        self._current_layer = 0

        layers = self._find_attention_layers(model)
        print(f"  Found {len(layers)} attention layers to hook.")

        for idx, attn_module in enumerate(layers):
            self.keys_per_layer.append([])
            self.vals_per_layer.append([])
            hook = attn_module.register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

    # ------------------------------------------------------------------
    @staticmethod
    def _find_attention_layers(model: nn.Module) -> list[nn.Module]:
        """Return the list of self_attn sub-modules, one per transformer layer."""
        layers = []
        # Qwen2 model: model.model.layers[i].self_attn
        for layer in model.model.layers:
            layers.append(layer.self_attn)
        return layers

    # ------------------------------------------------------------------
    def _make_hook(self, layer_idx: int):
        """Closure that captures layer_idx and appends K/V to the right bucket."""
        def hook(module, inputs, outputs):
            # outputs is a tuple: (attn_output, attn_weights?, past_key_value?)
            # For Qwen2 with output_attentions=False and use_cache=False the
            # module stores key_states / value_states as instance attributes
            # right before returning — we read them directly from the module.
            k = module.key_states    # shape: (B, num_kv_heads, T, head_dim)
            v = module.value_states  # shape: (B, num_kv_heads, T, head_dim)
            # Detach, move to CPU float32 for lossless storage
            self.keys_per_layer[layer_idx].append(k.detach().cpu().float())
            self.vals_per_layer[layer_idx].append(v.detach().cpu().float())
        return hook

    # ------------------------------------------------------------------
    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    def finalize(self) -> dict[str, torch.Tensor]:
        """
        Stack sample tensors into a single array per layer then return
        a dict with shapes [L, N, H, T, D].
        """
        # Each element of keys_per_layer[l] is (1, H, T, D) — squeeze batch dim
        keys = []
        vals = []
        for l_keys, l_vals in zip(self.keys_per_layer, self.vals_per_layer):
            # Stack along new sample dimension → (N, H, T, D)
            keys.append(torch.stack([t.squeeze(0) for t in l_keys], dim=0))
            vals.append(torch.stack([t.squeeze(0) for t in l_vals], dim=0))

        return {
            "keys":   torch.stack(keys, dim=0),   # (L, N, H, T, D)
            "values": torch.stack(vals, dim=0),   # (L, N, H, T, D)
        }


# ── helper: load one model ────────────────────────────────────────────────────

def load_model_and_tokenizer(model_id: str):
    print(f"\n[Load] {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # padding side must be left for decoder-only models
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=DTYPE,
        device_map="cpu",          # explicit CPU placement
        low_cpu_mem_usage=True,    # load shard-by-shard to keep peak RAM low
    )
    model.eval()
    print(f"  Config  : {model.config.num_hidden_layers} layers, "
          f"{model.config.num_key_value_heads} KV heads, "
          f"head_dim={model.config.hidden_size // model.config.num_attention_heads}")
    return model, tokenizer


# ── helper: patch attention modules to expose key/value states ───────────────

def patch_qwen_attention(model: nn.Module):
    """
    Qwen2Attention.forward() normally does not store key_states as an
    attribute.  We monkey-patch each layer's forward() with a thin wrapper
    that saves the tensors so the hook can read them.
    """
    import types

    def _patched_forward(self_attn, *args, **kwargs):
        # Call the original forward
        output = self_attn._original_forward(*args, **kwargs)
        # key_states / value_states were computed inside; we need to re-expose
        # them.  Easier: re-run just the projection for inspection.
        # Actually the cleanest approach for Qwen2 is to override at a lower
        # level.  We'll use a different strategy: wrap forward to intercept
        # the intermediate tensors.
        return output

    # Simpler strategy: override the module to cache intermediates.
    for layer in model.model.layers:
        attn = layer.self_attn

        original_forward = attn.forward

        def make_wrapper(orig_fwd, attn_ref):
            def wrapper(*args, **kwargs):
                # Force use_cache=False so no past_key_values complicates shapes
                kwargs["use_cache"] = False
                result = orig_fwd(*args, **kwargs)
                return result
            return wrapper

        attn.forward = make_wrapper(original_forward, attn)


# ── helper: run calibration and collect KV ───────────────────────────────────

def extract_kv(model: nn.Module, tokenizer, texts: list[str]) -> dict:
    """
    Tokenise each text, run a forward pass, and return the stacked KV dict.
    Uses a KVHookManager that reads key_states / value_states off each
    Qwen2Attention module after patching.
    """
    # Patch each attention layer's forward to cache its K/V as attributes
    _install_kv_attribute_patch(model)

    hook_mgr = KVHookManager(model)

    for i, text in enumerate(texts):
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",   # pad every sequence to MAX_SEQ_LEN so T is identical across samples
        )
        input_ids      = tokens["input_ids"].to(DEVICE)
        attention_mask = tokens["attention_mask"].to(DEVICE)

        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,          # we don't need past_key_values
            output_attentions=False,
        )

        if (i + 1) % 5 == 0:
            print(f"    Processed {i+1}/{len(texts)} samples …")

    hook_mgr.remove_hooks()
    kv = hook_mgr.finalize()
    return kv


def _install_kv_attribute_patch(model: nn.Module):
    """
    Monkey-patch each Qwen2Attention.forward so that, after the Q/K/V
    projections, key_states and value_states are stored as module attributes
    (self.key_states, self.value_states).  The hook in KVHookManager then
    reads them safely.

    This avoids rewriting the entire attention forward; we only wrap it.
    """
    # num_key_value_heads lives on the config, not on the attention module
    # in newer versions of transformers — read it once here and close over it.
    num_kv_heads = model.config.num_key_value_heads

    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, "_kv_patch_installed"):
            continue

        orig     = attn.forward
        head_dim = attn.head_dim   # this attribute is present in all versions

        def _wrap(orig_fn, mod, _num_kv_heads, _head_dim):
            def _forward(*args, **kwargs):
                kwargs.setdefault("use_cache", False)

                # ---- intercept via temporary inner hooks on the k/v proj ----
                captured = {}

                def _k_hook(m, inp, out):
                    captured["k_raw"] = out

                def _v_hook(m, inp, out):
                    captured["v_raw"] = out

                hk = mod.k_proj.register_forward_hook(_k_hook)
                hv = mod.v_proj.register_forward_hook(_v_hook)

                result = orig_fn(*args, **kwargs)

                hk.remove()
                hv.remove()

                # Reshape flat projection output → (B, num_kv_heads, T, head_dim)
                if "k_raw" in captured:
                    k_raw = captured["k_raw"]   # (B, T, num_kv_heads * head_dim)
                    v_raw = captured["v_raw"]
                    B, T, _ = k_raw.shape
                    mod.key_states   = k_raw.view(B, T, _num_kv_heads, _head_dim).transpose(1, 2)
                    mod.value_states = v_raw.view(B, T, _num_kv_heads, _head_dim).transpose(1, 2)

                return result
            return _forward

        attn.forward = _wrap(orig, attn, num_kv_heads, head_dim)
        attn._kv_patch_installed = True


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("outputs", exist_ok=True)

    print("=" * 60)
    print("  Stage 1 — Loading models")
    print("=" * 60)

    src_model, src_tok = load_model_and_tokenizer(SOURCE_MODEL_ID)
    tgt_model, tgt_tok = load_model_and_tokenizer(TARGET_MODEL_ID)

    print("\n" + "=" * 60)
    print("  Stage 2 — Extracting KV caches (calibration prompts)")
    print("=" * 60)

    print(f"\n[Source] Running {len(CALIBRATION_TEXTS)} prompts through {SOURCE_MODEL_ID}")
    kv_source = extract_kv(src_model, src_tok, CALIBRATION_TEXTS)

    print(f"\n[Target] Running {len(CALIBRATION_TEXTS)} prompts through {TARGET_MODEL_ID}")
    kv_target = extract_kv(tgt_model, tgt_tok, CALIBRATION_TEXTS)

    # ── print shapes ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Shape Report  (L=layers, N=samples, H=KV-heads, T=tokens, D=head_dim)")
    print("=" * 60)
    for name, kv in [("Source (0.5B)", kv_source), ("Target (1.5B)", kv_target)]:
        K = kv["keys"]
        V = kv["values"]
        print(f"\n  {name}")
        print(f"    keys   shape : {tuple(K.shape)}  — (L, N, H, T, D)")
        print(f"    values shape : {tuple(V.shape)}  — (L, N, H, T, D)")
        print(f"    dtype        : {K.dtype}")
        print(f"    memory (keys): {K.nbytes / 1e6:.1f} MB")

    # ── save to disk ──────────────────────────────────────────────────
    src_path = "outputs/kv_source.pt"
    tgt_path = "outputs/kv_target.pt"
    torch.save(kv_source, src_path)
    torch.save(kv_target, tgt_path)
    print(f"\n[Saved] {src_path}")
    print(f"[Saved] {tgt_path}")
    print("\nDone. Stages 1 & 2 complete.")


if __name__ == "__main__":
    main()
