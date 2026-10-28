"""
KV cache hooks for "Back from the Future: Key-Value Cache Management by Counter-Causal Surprise."
Copyright (c) 2026, Metacognition (http://metacognitionai.com)

Public API
----------
CacheState                  — dataclass bundling (kv, tok, pos, hidden)
generate_with_kv_hook       — autoregressive generation with an optional hook
sliding_window_hook         — keep the most recent window_size tokens
importance_eviction_hook    — evict by last-layer key similarity
h2o_eviction_hook           — Heavy-Hitter Oracle (Zhang et al., 2023)
counter_causal_hook         — counter-causal surprise, full L-layer pass (ours)
counter_causal_fast_hook    — counter-causal surprise, single-layer approx (ours)
"""

from dataclasses import dataclass
from typing import Optional, Callable, Literal

import torch
import torch.nn.functional as F
from transformers import DynamicCache
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Cache state
# ---------------------------------------------------------------------------

@dataclass
class CacheState:
    """All mutable state associated with the KV cache at one point in time."""
    kv:     DynamicCache
    tok:    torch.Tensor                    # [1, seq_len]  int token ids
    pos:    torch.Tensor                    # [1, seq_len]  int position ids
    hidden: Optional[torch.Tensor] = None   # [1, seq_len, d_model]  h^{(L-1)}


# ---------------------------------------------------------------------------
# Cache utilities
# ---------------------------------------------------------------------------

def _get_cache_size(state: CacheState) -> int:
    if isinstance(state.kv, tuple):
        return max(kv[0].shape[2] for kv in state.kv)
    return max(layer.keys.shape[2] for layer in state.kv.layers)


def _select_tokens(state: CacheState, frozen_size: int, keep_indices: torch.Tensor) -> CacheState:
    """Return a new CacheState keeping the first `frozen_size` tokens plus `keep_indices`."""
    new_kv = DynamicCache()
    if isinstance(state.kv, tuple):
        for layer_idx, (k, v) in enumerate(state.kv):
            k_new = torch.cat((k[:, :, :frozen_size, :], k[:, :, keep_indices + frozen_size, :]), dim=2)
            v_new = torch.cat((v[:, :, :frozen_size, :], v[:, :, keep_indices + frozen_size, :]), dim=2)
            new_kv.update(k_new, v_new, layer_idx)
    else:
        for layer_idx, layer in enumerate(state.kv.layers):
            k_new = torch.cat((layer.keys[:, :, :frozen_size, :], layer.keys[:, :, keep_indices + frozen_size, :]), dim=2)
            v_new = torch.cat((layer.values[:, :, :frozen_size, :], layer.values[:, :, keep_indices + frozen_size, :]), dim=2)
            new_kv.update(k_new, v_new, layer_idx)

    new_tok = torch.cat((state.tok[:, :frozen_size], state.tok[:, keep_indices + frozen_size]), dim=1)
    new_pos = torch.cat((state.pos[:, :frozen_size], state.pos[:, keep_indices + frozen_size]), dim=1)
    new_hidden = None
    if state.hidden is not None:
        new_hidden = torch.cat(
            (state.hidden[:, :frozen_size, :], state.hidden[:, keep_indices + frozen_size, :]), dim=1
        )
    return CacheState(kv=new_kv, tok=new_tok, pos=new_pos, hidden=new_hidden)


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to q. cos/sin shape: [batch, seq_len, head_dim]."""
    cos = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin)


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

def _eos_ids(model, tokenizer) -> frozenset:
    """The token ids that end generation, as transformers' generate takes them: the model's
    generation_config.eos_token_id (Llama-3.1-Instruct: <|end_of_text|>, <|eom_id|>, <|eot_id|>)
    and the tokenizer's eos token."""
    ids = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    ids = set() if ids is None else ({ids} if isinstance(ids, int) else set(ids))
    if getattr(tokenizer, "eos_token_id", None) is not None:
        ids.add(tokenizer.eos_token_id)
    return frozenset(ids)


def generate_with_kv_hook(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 16,
    chunk_size: int = 4096,
    kv_hook: Optional[Callable[[CacheState], CacheState]] = None,
    stop_fn: Optional[Callable] = None,
    refresh_mode: Literal['chunked', 'prefill_end'] = 'chunked',
) -> str:
    """
    Autoregressive generation with an optional KV cache hook.

    Args:
        model:          HuggingFace causal LM.
        tokenizer:      Corresponding tokenizer (used for its eos token and decoding).
        input_ids:      Prompt token ids, shape [1, prompt_len].
        max_new_tokens: Maximum tokens to generate.
        chunk_size:     Tokens processed between hook invocations.
        kv_hook:        CacheState -> CacheState; evicts/compresses the cache.
        stop_fn:        List[int] -> bool; stops generation when it returns True.
        refresh_mode:
            'chunked'     — hook fires every chunk_size tokens (prefill + decode).
            'prefill_end' — hook fires once after all prompt tokens are processed.

    A forward hook on model.model.layers[-2] captures h^{(L-1)} into state.hidden
    at every forward pass; this is consumed by counter_causal_fast_hook.
    """
    _hidden_buf = [None]

    def _capture(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        _hidden_buf[0] = h.detach()

    _hook_handle = model.model.layers[-2].register_forward_hook(_capture)

    try:
        # ---- initial prefill chunk ----
        first_chunk = input_ids[:, :chunk_size]
        with torch.no_grad():
            out = model(input_ids=first_chunk, use_cache=True)

        state = CacheState(
            kv=out.past_key_values,
            tok=first_chunk,
            pos=torch.arange(0, first_chunk.shape[1], device=model.device, dtype=torch.long).unsqueeze(0),
            hidden=_hidden_buf[0],
        )

        if kv_hook and refresh_mode == 'chunked' and input_ids.shape[1] > chunk_size:
            state = kv_hook(state)

        # ---- remaining prefill chunks ----
        for step in tqdm(range(chunk_size, input_ids.shape[1], chunk_size), desc="Prefill", leave=False):
            chunk = input_ids[:, step:step + chunk_size]
            pos_ids = torch.arange(step, step + chunk.shape[1], device=model.device, dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                out = model(input_ids=chunk, past_key_values=state.kv, position_ids=pos_ids, use_cache=True)
            state = CacheState(
                kv=out.past_key_values,
                tok=torch.cat((state.tok, chunk), dim=1),
                pos=torch.cat((state.pos, pos_ids), dim=1),
                hidden=torch.cat([state.hidden, _hidden_buf[0]], dim=1),
            )
            if kv_hook and refresh_mode == 'chunked':
                state = kv_hook(state)

        if kv_hook and refresh_mode == 'prefill_end':
            state = kv_hook(state)

        # ---- decode ----
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        current_pos = input_ids.shape[1]
        generated = [next_token.item()]
        eos = _eos_ids(model, tokenizer)

        for step in tqdm(range(1, max_new_tokens), desc="Decode", leave=False):
            if generated[-1] in eos:
                break
            pos_ids = torch.tensor([[current_pos]], device=model.device, dtype=torch.long)
            with torch.no_grad():
                out = model(input_ids=next_token, past_key_values=state.kv, position_ids=pos_ids, use_cache=True)
            current_pos += 1
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tok = next_token.item()
            generated.append(tok)

            state = CacheState(
                kv=out.past_key_values,
                tok=torch.cat((state.tok, next_token), dim=1),
                pos=torch.cat((state.pos, pos_ids), dim=1),
                hidden=torch.cat([state.hidden, _hidden_buf[0]], dim=1),
            )
            if kv_hook and refresh_mode == 'chunked' and (input_ids.shape[1] + step) % chunk_size == 0:
                state = kv_hook(state)

            if stop_fn and stop_fn(generated):
                break

    finally:
        _hook_handle.remove()

    return tokenizer.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Baseline hooks
# ---------------------------------------------------------------------------

def sliding_window_hook(window_size: int = 2048, frozen_size: int = 0):
    """Keep only the most recent `window_size` tokens (plus the first `frozen_size`)."""
    def hook(state: CacheState) -> CacheState:
        if _get_cache_size(state) <= window_size + frozen_size:
            return state
        body_size = state.tok.shape[1] - frozen_size
        keep = torch.arange(body_size - window_size, body_size, device=state.tok.device)
        return _select_tokens(state, frozen_size, keep)
    return hook


def importance_eviction_hook(cache_size: int = 2048, frozen_size: int = 0):
    """Evict the lowest-importance tokens using the last layer's key vectors as a proxy."""
    def hook(state: CacheState) -> CacheState:
        if _get_cache_size(state) <= cache_size + frozen_size:
            return state
        if isinstance(state.kv, tuple):
            k_last = state.kv[-1][0][:, :, frozen_size:, :]
        else:
            k_last = state.kv.layers[-1].keys[:, :, frozen_size:, :]
        k_last = k_last / torch.clamp(k_last.abs().max(), min=1e-6)
        q_last = k_last[:, :, -1:, :]
        scores = (q_last @ k_last.transpose(-2, -1)).squeeze(-2).mean(1)
        keep = scores.topk(cache_size, dim=-1).indices[0].sort().values
        return _select_tokens(state, frozen_size, keep)
    return hook


def h2o_eviction_hook(cache_size: int = 2048, recent_size: int = 512, frozen_size: int = 0):
    """H2O (Heavy-Hitter Oracle) KV cache eviction (Zhang et al., arXiv:2306.14048)."""
    heavy_size = cache_size - recent_size
    assert heavy_size > 0, "recent_size must be < cache_size"
    accum_scores = [None]

    def hook(state: CacheState) -> CacheState:
        body_size = _get_cache_size(state) - frozen_size
        scores_sum = None
        n_layers = 0
        layers = state.kv if isinstance(state.kv, tuple) else state.kv.layers
        for layer in layers:
            k = layer[0] if isinstance(layer, tuple) else layer.keys
            k_body = k[:, :, frozen_size:, :]
            n_query = min(recent_size, body_size)
            q = k_body[:, :, -n_query:, :]
            scale = k_body.shape[-1] ** -0.5
            attn = torch.softmax((q @ k_body.transpose(-2, -1)) * scale, dim=-1)
            layer_scores = attn.sum(dim=2).mean(dim=(0, 1))
            scores_sum = layer_scores if scores_sum is None else scores_sum + layer_scores
            n_layers += 1
        scores_sum = scores_sum / n_layers

        if accum_scores[0] is None or accum_scores[0].shape[0] > body_size:
            accum_scores[0] = scores_sum
        else:
            prev = accum_scores[0].shape[0]
            if prev < body_size:
                pad = torch.zeros(body_size - prev, device=scores_sum.device, dtype=scores_sum.dtype)
                accum_scores[0] = torch.cat([accum_scores[0], pad])
            accum_scores[0] = accum_scores[0] + scores_sum

        if body_size <= cache_size:
            return state

        non_recent = body_size - recent_size
        heavy_idx = accum_scores[0][:non_recent].topk(min(heavy_size, non_recent)).indices.sort().values
        recent_idx = torch.arange(body_size - recent_size, body_size, device=state.tok.device)
        keep = torch.cat([heavy_idx, recent_idx])
        accum_scores[0] = accum_scores[0][keep]
        return _select_tokens(state, frozen_size, keep)
    return hook


# ---------------------------------------------------------------------------
# Counter-causal hooks (ours)
# ---------------------------------------------------------------------------

def counter_causal_hook(model, cache_size: int = 2048, frozen_size: int = 0,
                                recent_size: int = 0, min_gap: int = 0):
    """
    Counter-causal surprise eviction: full L-layer pass reusing the existing cached K/V.

    Runs the model on the original token sequence with an upper-triangular (counter-causal)
    attention mask so that position i attends only to positions j > i + min_gap. The K and V
    tensors already in the cache are reused at every layer. Only Q is freshly projected.
    The FFN at each layer is applied normally.

    The surprise score at position i is the logit of x_i given only future context:
        score(i) = logit(x_i | {K_l(j), V_l(j) : j > i + min_gap, all layers l})

    Tokens with the lowest surprise scores are most predictable from future tokens and are
    therefore the best candidates for eviction.

    Args:
        model:       HuggingFace causal LM (must match the one used for generation).
        cache_size:  Maximum number of non-frozen tokens to retain.
        frozen_size: Number of leading tokens never evicted (e.g. system prompt).
        recent_size: Number of trailing tokens always kept without scoring.
        min_gap:     Block attention to the nearest min_gap neighbours (reduces K/V leakage).
    """
    assert recent_size < cache_size
    surprise_size = cache_size - recent_size

    def hook(state: CacheState) -> CacheState:
        if _get_cache_size(state) <= cache_size + frozen_size:
            return state

        seq_len = state.tok.shape[1]
        body_size = seq_len - frozen_size
        device = state.tok.device

        cfg = model.config
        num_heads = cfg.num_attention_heads
        num_kv_heads = cfg.num_key_value_heads

        cc_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1 + min_gap
        )

        with torch.no_grad():
            hidden = model.model.embed_tokens(state.tok)

            for layer_idx, layer in enumerate(model.model.layers):
                if isinstance(state.kv, tuple):
                    k = state.kv[layer_idx][0]
                    v = state.kv[layer_idx][1]
                else:
                    k = state.kv.layers[layer_idx].keys
                    v = state.kv.layers[layer_idx].values

                attn = layer.self_attn
                head_dim = attn.head_dim

                h_norm = layer.input_layernorm(hidden)
                q = attn.q_proj(h_norm)
                q = q.view(1, seq_len, num_heads, head_dim).transpose(1, 2)

                cos, sin = model.model.rotary_emb(h_norm, position_ids=state.pos)
                q = _apply_rope(q, cos, sin)

                if num_kv_heads < num_heads:
                    g = num_heads // num_kv_heads
                    k = k.repeat_interleave(g, dim=1)
                    v = v.repeat_interleave(g, dim=1)

                attn_out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=cc_mask.unsqueeze(0).unsqueeze(0),
                    scale=head_dim ** -0.5,
                )
                attn_out = torch.nan_to_num(attn_out, nan=0.0)
                attn_out = attn_out.transpose(1, 2).reshape(1, seq_len, -1)
                attn_out = attn.o_proj(attn_out)

                hidden = hidden + attn_out
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))

            logits = model.lm_head(model.model.norm(hidden)).float()

        scores = torch.gather(logits, 2, state.tok.unsqueeze(-1)).squeeze(-1)[0]
        scores[-(min_gap + 1):] = float('-inf')

        non_recent_size = body_size - recent_size
        body_scores = scores[frozen_size:frozen_size + non_recent_size]
        surprise_indices = body_scores.topk(min(surprise_size, non_recent_size), largest=False).indices
        recent_indices = torch.arange(body_size - recent_size, body_size, device=device)
        keep = torch.cat([surprise_indices, recent_indices]).sort().values
        return _select_tokens(state, frozen_size, keep)

    return hook


def counter_causal_fast_hook(model, cache_size: int = 2048, frozen_size: int = 0,
                              recent_size: int = 0, min_gap: int = 0):
    """
    Counter-causal surprise eviction: single last-layer approximation.

    Instead of running all L layers with a counter-causal mask, this hook:
      1. Takes h^{(L-1)} (the output of the second-to-last layer) from state.hidden.
         generate_with_kv_hook captures this automatically via a registered forward hook.
      2. Projects Q from h^{(L-1)} via the last layer's q_proj + RoPE.
      3. Reuses the last layer's cached K and V (already carry RoPE from the forward pass).
      4. Applies a counter-causal attention mask (position i attends to j > i + min_gap).
      5. Applies o_proj, skips the FFN (approximation), then applies the final norm + LM head.
      6. Uses the logit of the actual token at each position as the surprise score.

    Args:
        model:       HuggingFace causal LM (must match the one used for generation).
        cache_size:  Maximum number of non-frozen tokens to retain.
        frozen_size: Number of leading tokens never evicted (e.g. system prompt).
        recent_size: Number of trailing tokens always kept without scoring.
        min_gap:     Block attention to the nearest min_gap neighbours (reduces K/V leakage).
    """
    assert recent_size < cache_size
    surprise_size = cache_size - recent_size

    def hook(state: CacheState) -> CacheState:
        if _get_cache_size(state) <= cache_size + frozen_size:
            return state
        if state.hidden is None:
            raise RuntimeError(
                "counter_causal_fast_hook requires state.hidden. "
                "Use generate_with_kv_hook from hooks.py — it registers the capture hook automatically."
            )

        seq_len = state.tok.shape[1]
        body_size = seq_len - frozen_size
        device = state.tok.device

        last_layer = model.model.layers[-1]
        last_attn = last_layer.self_attn
        cfg = model.config
        num_heads = cfg.num_attention_heads
        num_kv_heads = cfg.num_key_value_heads
        head_dim = last_attn.head_dim

        with torch.no_grad():
            h_norm = last_layer.input_layernorm(state.hidden)

            q = last_attn.q_proj(h_norm)
            q = q.view(1, seq_len, num_heads, head_dim).transpose(1, 2)

            cos, sin = model.model.rotary_emb(h_norm, position_ids=state.pos)
            q = _apply_rope(q, cos, sin)

            if isinstance(state.kv, tuple):
                k = state.kv[-1][0]
                v = state.kv[-1][1]
            else:
                k = state.kv.layers[-1].keys
                v = state.kv.layers[-1].values

            if num_kv_heads < num_heads:
                g = num_heads // num_kv_heads
                k = k.repeat_interleave(g, dim=1)
                v = v.repeat_interleave(g, dim=1)

            cc_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1 + min_gap
            )

            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=cc_mask.unsqueeze(0).unsqueeze(0),
                scale=head_dim ** -0.5,
            )
            attn_out = torch.nan_to_num(attn_out, nan=0.0)
            attn_out = attn_out.transpose(1, 2).reshape(1, seq_len, -1)
            attn_out = last_attn.o_proj(attn_out)

            h_mid = state.hidden + attn_out  # skip FFN
            logits = model.lm_head(model.model.norm(h_mid))

        scores = torch.gather(logits, 2, state.tok.unsqueeze(-1)).squeeze(-1)[0]
        scores[-(min_gap + 1):] = float('-inf')

        non_recent_size = body_size - recent_size
        body_scores = scores[frozen_size:frozen_size + non_recent_size]
        surprise_indices = body_scores.topk(min(surprise_size, non_recent_size), largest=False).indices
        recent_indices = torch.arange(body_size - recent_size, body_size, device=device)
        keep = torch.cat([surprise_indices, recent_indices]).sort().values
        return _select_tokens(state, frozen_size, keep)

    return hook
