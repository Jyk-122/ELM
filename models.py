import os
import sys
import json
import torch

from convert_weights import hf_to_meta
from transformers import AutoModelForCausalLM


def LLMs_Lora(args, **kwargs):
    from elm.model_lora import ModelArgs, Transformer

    hf_model_path = args.hf_model_path
    
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        torch_dtype='bfloat16',
        low_cpu_mem_usage=True,
        device_map='cpu',
        trust_remote_code=True,
    )
    
    converted_state_dict, converted_model_args = hf_to_meta(model.state_dict(), model.config)

    if "Llama" in hf_model_path:
        converted_model_args.update({"model_type": "llama"})
    if "Qwen" in hf_model_path:
        converted_model_args.update({"model_type": "qwen"})

    model_args: ModelArgs = ModelArgs(
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        **converted_model_args
    )
    
    print(model_args)

    if torch.cuda.is_bf16_supported():
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    else:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    
    model_lora = Transformer(model_args)
    model_lora.load_state_dict(converted_state_dict, strict=False)

    # only tune the lora module
    for name, param in model_lora.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    return model_lora


def ELMs(args, **kwargs):
    from elm.model_elm import ModelArgs, Transformer

    hf_model_path = args.hf_model_path
    
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        torch_dtype='bfloat16',
        low_cpu_mem_usage=True,
        device_map='cpu',
        trust_remote_code=True,
    )
    
    converted_state_dict, converted_model_args = hf_to_meta(model.state_dict(), model.config)

    if "Llama" in hf_model_path:
        converted_model_args.update({"model_type": "llama"})
    if "Qwen" in hf_model_path:
        converted_model_args.update({"model_type": "qwen"})
    
    ### prune layer interval setting
    if args.prune_interval_layer_stats is not None:
        stats = torch.load(args.prune_interval_layer_stats, map_location='cpu')
        prune_interval_start_layer = stats["model"]["policy.weight"].argmax().item()
    else:
        prune_interval_start_layer = args.prune_interval_start_layer
    
    model_args: ModelArgs = ModelArgs(
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        prune_interval_start_layer=prune_interval_start_layer,
        prune_interval_len=args.prune_interval_len,
        **converted_model_args
    )
    
    print(model_args)

    if torch.cuda.is_bf16_supported():
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    else:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    
    model_elm = Transformer(model_args)
    model_elm.load_state_dict(converted_state_dict, strict=False)

    # Merge LoRA weights into Transformer.
    if args.lora_model_path is not None:
        lora_checkpoint = torch.load(args.lora_model_path, map_location="cpu")
        state = lora_checkpoint["model"]
        for i in range(model_args.n_layers):
            loraq_weight = state[f"layers.{i}.attention.lora_q.linear_A.weight"].T @ state[f"layers.{i}.attention.lora_q.linear_B.weight"].T
            lorak_weight = state[f"layers.{i}.attention.lora_k.linear_A.weight"].T @ state[f"layers.{i}.attention.lora_k.linear_B.weight"].T
            lorav_weight = state[f"layers.{i}.attention.lora_v.linear_A.weight"].T @ state[f"layers.{i}.attention.lora_v.linear_B.weight"].T
            lorao_weight = state[f"layers.{i}.attention.lora_o.linear_A.weight"].T @ state[f"layers.{i}.attention.lora_o.linear_B.weight"].T
            lora1_weight = state[f"layers.{i}.feed_forward.lora_1.linear_A.weight"].T @ state[f"layers.{i}.feed_forward.lora_1.linear_B.weight"].T
            lora2_weight = state[f"layers.{i}.feed_forward.lora_2.linear_A.weight"].T @ state[f"layers.{i}.feed_forward.lora_2.linear_B.weight"].T
            lora3_weight = state[f"layers.{i}.feed_forward.lora_3.linear_A.weight"].T @ state[f"layers.{i}.feed_forward.lora_3.linear_B.weight"].T

            device = model_elm.layers[i].attention.wq.weight.device

            model_elm.layers[i].attention.wq.weight.data += loraq_weight.data.T.to(device)
            model_elm.layers[i].attention.wk.weight.data += lorak_weight.data.T.to(device)
            model_elm.layers[i].attention.wv.weight.data += lorav_weight.data.T.to(device)
            model_elm.layers[i].attention.wo.weight.data += lorao_weight.data.T.to(device)
            model_elm.layers[i].feed_forward.w1.weight.data += lora1_weight.data.T.to(device)
            model_elm.layers[i].feed_forward.w2.weight.data += lora2_weight.data.T.to(device)
            model_elm.layers[i].feed_forward.w3.weight.data += lora3_weight.data.T.to(device)

    # init the streamline module
    source_layer = model_elm.layers[prune_interval_start_layer]
    source_layer_state = source_layer.state_dict()
    model_elm.prune_interval_replace_module.load_state_dict(source_layer_state, strict=False)

    # only tune the streamline block
    for name, param in model_elm.named_parameters():
        if "prune_interval" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    return model_elm


def ELMs_GPPO(args, **kwargs):
    from elm.model_gppo import ModelArgs, GPPO

    hf_model_path = args.hf_model_path
    
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        torch_dtype='bfloat16',
        low_cpu_mem_usage=True,
        device_map='cpu',
        trust_remote_code=True,
    )
    
    converted_state_dict, converted_model_args = hf_to_meta(model.state_dict(), model.config)

    if "Llama" in hf_model_path:
        converted_model_args.update({"model_type": "llama"})
    if "Qwen" in hf_model_path:
        converted_model_args.update({"model_type": "qwen"})
    
    ### prune layer interval setting
    if args.prune_interval_layer_stats is not None:
        stats = torch.load(args.prune_interval_layer_stats, map_location='cpu')
        prune_interval_start_layer = stats["model"]["policy.weight"].argmax().item()
    else:
        prune_interval_start_layer = args.prune_interval_start_layer
    
    model_args: ModelArgs = ModelArgs(
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        prune_interval_start_layer=prune_interval_start_layer,
        prune_interval_len=args.prune_interval_len,
        **converted_model_args
    )
    
    print(model_args)

    if torch.cuda.is_bf16_supported():
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    else:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    
    model = GPPO(model_args)
    model_elm = model.reward_model
    model_elm.load_state_dict(converted_state_dict, strict=False)

    # Merge LoRA weights into Transformer.
    if args.lora_model_path is not None:
        lora_checkpoint = torch.load(args.lora_model_path, map_location="cpu")
        state = lora_checkpoint["model"]
        for i in range(model_args.n_layers):
            loraq_weight = state[f"layers.{i}.attention.lora_q.linear_A.weight"].T @ state[f"layers.{i}.attention.lora_q.linear_B.weight"].T
            lorak_weight = state[f"layers.{i}.attention.lora_k.linear_A.weight"].T @ state[f"layers.{i}.attention.lora_k.linear_B.weight"].T
            lorav_weight = state[f"layers.{i}.attention.lora_v.linear_A.weight"].T @ state[f"layers.{i}.attention.lora_v.linear_B.weight"].T
            lorao_weight = state[f"layers.{i}.attention.lora_o.linear_A.weight"].T @ state[f"layers.{i}.attention.lora_o.linear_B.weight"].T
            lora1_weight = state[f"layers.{i}.feed_forward.lora_1.linear_A.weight"].T @ state[f"layers.{i}.feed_forward.lora_1.linear_B.weight"].T
            lora2_weight = state[f"layers.{i}.feed_forward.lora_2.linear_A.weight"].T @ state[f"layers.{i}.feed_forward.lora_2.linear_B.weight"].T
            lora3_weight = state[f"layers.{i}.feed_forward.lora_3.linear_A.weight"].T @ state[f"layers.{i}.feed_forward.lora_3.linear_B.weight"].T

            device = model_elm.layers[i].attention.wq.weight.device

            model_elm.layers[i].attention.wq.weight.data += loraq_weight.data.T.to(device)
            model_elm.layers[i].attention.wk.weight.data += lorak_weight.data.T.to(device)
            model_elm.layers[i].attention.wv.weight.data += lorav_weight.data.T.to(device)
            model_elm.layers[i].attention.wo.weight.data += lorao_weight.data.T.to(device)
            model_elm.layers[i].feed_forward.w1.weight.data += lora1_weight.data.T.to(device)
            model_elm.layers[i].feed_forward.w2.weight.data += lora2_weight.data.T.to(device)
            model_elm.layers[i].feed_forward.w3.weight.data += lora3_weight.data.T.to(device)

    # only tune the policy model and lora modules in reward model
    for name, param in model.named_parameters():
        if "policy_model" in name or "lora" in name or "adaptor" in name:
            param.requires_grad = True
            param.data = param.data.float()
        else:
            param.requires_grad = False
    
    return model