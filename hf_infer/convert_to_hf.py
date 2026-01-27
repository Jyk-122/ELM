import sys
import json
import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("./")
from elm.model import ModelArgs, Transformer
from convert_weights import _FROM_HF_WEIGHT, get_mapped_key

from modeling_qwen.modeling_qwen2_elm import Qwen2ELMForCausalLM
from modeling_qwen.configuration_qwen2_elm import Qwen2ELMConfig


def convert(
    base_model_path: str,
    base_lora_path: str,
    elm_ckpt_path: str,
    elm_args_path: str,
    model_elm_save_path: str,
):
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    with open(elm_args_path, "r") as f:
        params = json.loads(f.read())

    prune_interval_start_layer = params["prune_interval_start_layer"]
    prune_interval_len = params["prune_interval_len"]

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype='float',
        low_cpu_mem_usage=True,
        device_map='cpu',
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    config = model.config

    # Lora load and merge

    lora_ckpt = torch.load(base_lora_path, map_location='cpu')
    state = lora_ckpt['model']

    def _permute(t, n_head):
        return (
            t.reshape(n_head, t.shape[0] // n_head // 2, 2, *t.shape[1:])
            .transpose(1, 2)
            .reshape(*t.shape)
        )
    
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads

    for i in range(params["n_layers"]):
        loraq_weight = (state[f"layers.{i}.attention.lora_q.linear_A.weight"].T @ state[f"layers.{i}.attention.lora_q.linear_B.weight"].T).T
        lorak_weight = (state[f"layers.{i}.attention.lora_k.linear_A.weight"].T @ state[f"layers.{i}.attention.lora_k.linear_B.weight"].T).T
        lorav_weight = (state[f"layers.{i}.attention.lora_v.linear_A.weight"].T @ state[f"layers.{i}.attention.lora_v.linear_B.weight"].T).T
        lorao_weight = (state[f"layers.{i}.attention.lora_o.linear_A.weight"].T @ state[f"layers.{i}.attention.lora_o.linear_B.weight"].T).T
        lora1_weight = (state[f"layers.{i}.feed_forward.lora_1.linear_A.weight"].T @ state[f"layers.{i}.feed_forward.lora_1.linear_B.weight"].T).T
        lora2_weight = (state[f"layers.{i}.feed_forward.lora_2.linear_A.weight"].T @ state[f"layers.{i}.feed_forward.lora_2.linear_B.weight"].T).T
        lora3_weight = (state[f"layers.{i}.feed_forward.lora_3.linear_A.weight"].T @ state[f"layers.{i}.feed_forward.lora_3.linear_B.weight"].T).T

        loraq_weight = _permute(loraq_weight, num_heads)
        lorak_weight = _permute(lorak_weight, num_kv_heads)

        model.model.layers[i].self_attn.q_proj.weight.data += loraq_weight.data
        model.model.layers[i].self_attn.k_proj.weight.data += lorak_weight.data
        model.model.layers[i].self_attn.v_proj.weight.data += lorav_weight.data
        model.model.layers[i].self_attn.o_proj.weight.data += lorao_weight.data
        model.model.layers[i].mlp.gate_proj.weight.data += lora1_weight.data
        model.model.layers[i].mlp.up_proj.weight.data += lora3_weight.data
        model.model.layers[i].mlp.down_proj.weight.data += lora2_weight.data
    
    print("Lora checkpoint adapted.")

    for idx in range(prune_interval_start_layer + 1, prune_interval_start_layer + prune_interval_len):
        model.model.layers.pop(prune_interval_start_layer + 1)

    elm_checkpoint = torch.load(elm_ckpt_path, map_location="cpu")
    state_dict = elm_checkpoint["model"]
    converted_state_dict = {}

    _FROM_META_WEIGHT = {}
    for key, value in _FROM_HF_WEIGHT.items():
        if "rotary_emb.inv_freq" not in key:
            value_ = value.replace("layers.{}", "prune_interval_replace_module")
            key_ = key.replace("layers.{}", f"layers.{prune_interval_start_layer}")
            _FROM_META_WEIGHT[value_] = key_

    _FROM_META_WEIGHT["prune_interval_replace_module.proj.weight"] = f"model.layers.{prune_interval_start_layer}.proj.weight"
    _FROM_META_WEIGHT["prune_interval_replace_module.proj.bias"] = f"model.layers.{prune_interval_start_layer}.proj.bias"

    for key, value in state_dict.items():
        if "rotary_emb.inv_freq" not in key:  # Skip loading the position embeddings
            new_key = get_mapped_key(key, _FROM_META_WEIGHT)
            if "wq" in key:
                value = _permute(value, num_heads)
            elif "wk" in key:
                value = _permute(value, num_kv_heads)
            converted_state_dict[new_key] = value

    config_elm = Qwen2ELMConfig.from_qwen2_config(
        config=config,
        fixed_point_layer_idx=prune_interval_start_layer,
        fixed_point_layer_interval_len=prune_interval_len,
        fixed_point_exit_threshold=1.0
    )
    model_elm = Qwen2ELMForCausalLM(config_elm)

    model_elm.load_state_dict(model.state_dict(), strict=False)
    model_elm.load_state_dict(converted_state_dict, strict=False)

    exit(0)

    # save
    model_elm.save_pretrained(model_elm_save_path, max_shard_size="10GB")
    tokenizer.save_pretrained(model_elm_save_path)
    config_elm.save_pretrained(model_elm_save_path)


if __name__ == "__main__":
    fire.Fire(convert)
