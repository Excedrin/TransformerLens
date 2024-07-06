import einops
import torch
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

def convert_glm_weights(glm_model, cfg: HookedTransformerConfig):
    state_dict = {}

    # Embedding weights
    state_dict["embed.W_E"] = glm_model.transformer.embedding.word_embeddings.weight

    # Transformer layers
    for l in range(cfg.n_layers):
        print("loading layer:", l)
        layer = glm_model.transformer.encoder.layers[l]

        # LayerNorm weights
        state_dict[f"blocks.{l}.ln1.w"] = layer.input_layernorm.weight

        # Attention weights (assuming split for query, key, and value)
        W_QKV = layer.self_attention.query_key_value.weight

        # Reshape and then split into Q, K, V
        W_QKV = einops.rearrange(W_QKV, "(n h) d -> n h d", n=cfg.n_heads)
        W_Q, W_K, W_V = W_QKV.chunk(3, dim=1)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        # Attention output projection
        state_dict[f"blocks.{l}.attn.W_O"] = layer.self_attention.dense.weight.T

        # Attention biases
        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn.b_K"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn.b_V"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        # MLP weights
        state_dict[f"blocks.{l}.mlp.W_in"] = layer.mlp.dense_h_to_4h.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = layer.mlp.dense_h_to_4h.bias
        state_dict[f"blocks.{l}.mlp.W_out"] = layer.mlp.dense_4h_to_h.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = layer.mlp.dense_4h_to_h.bias

        # Additional LayerNorm if present
        state_dict[f"blocks.{l}.ln2.w"] = layer.post_attention_layernorm.weight

    # Final LayerNorm if used
    if cfg.final_rms:
        state_dict["ln_final.w"] = glm_model.transformer.encoder.final_layernorm.weight

    # Output layer weights
    state_dict["unembed.W_U"] = glm_model.transformer.output_layer.weight

    print("returning state_dict")

    return state_dict
