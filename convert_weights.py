import re

from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# state dict key mappings from HF's format to meta's format
_FROM_HF_WEIGHT = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.wq.bias",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.wk.bias",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.wv.bias",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
}

_FROM_HF_MODELARGS = {
    "hidden_size": "dim",
    "num_hidden_layers": "n_layers",
    "num_attention_heads": "n_heads",
    "num_key_value_heads": "n_kv_heads",
    "vocab_size": "vocab_size",
    "intermediate_size": "intermediate_size",
    "rms_norm_eps": "norm_eps",
    "rope_theta": "rope_theta",
}


def get_mapped_key(key: str, mapping_dict: Dict[str, str]) -> str:
    try:
        if "layers" in key:
            # Replace layer number with "{}" to create key for lookup
            abstract_key = re.sub(r"(\.\d+)", ".{}", key)
            layer_num = re.search(r"\d+", key).group(0)
            new_key = mapping_dict[abstract_key]
            new_key = new_key.format(layer_num)
        else:
            new_key = mapping_dict[key]
    except KeyError as e:
        raise Exception(
            f'Error converting the state dict. Found unexpected key: "{key}". '
            "Please make sure you're loading a checkpoint with the right format. "
        ) from e

    return new_key


def hf_to_meta(
    state_dict: Dict[str, torch.Tensor],
    config: Dict,
    num_heads: int = 32,
    num_kv_heads: int = 32,
    dim: int = 4096,
    head_dim: int = None,
):
    if not isinstance(config, dict):
        config = config.to_dict()

    converted_state_dict = {}

    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    dim = config["hidden_size"]

    def _permute(t, n_head):
        return (
            t.reshape(n_head, 2, t.shape[0] // n_head // 2, *t.shape[1:])
            .transpose(1, 2)
            .reshape(*t.shape)
        )

    for key, value in state_dict.items():
        if "rotary_emb.inv_freq" not in key:  # Skip loading the position embeddings
            new_key = get_mapped_key(key, _FROM_HF_WEIGHT)
            if "q_proj" in key:
                value = _permute(value, num_heads)
            elif "k_proj" in key:
                value = _permute(value, num_kv_heads)

            converted_state_dict[new_key] = value

    converted_modelargs = {}
    
    for key, value in config.items():
        if key not in _FROM_HF_MODELARGS.keys():
            continue
        new_key = _FROM_HF_MODELARGS[key]
        converted_modelargs[new_key] = value

    return converted_state_dict, converted_modelargs
