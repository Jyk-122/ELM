# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("../")
from elm.model import ModelArgs, Transformer
from convert_weights import hf_to_meta


class ELM:
    @staticmethod
    def build(
        hf_model_path: str,
        lora_model_path: Optional[str],
        elm_ckpt_path: str,
        model_args_path: str,
        max_seq_len: int,
        max_batch_size: int,
        fixed_point_solver: str,
        fixed_point_max_iters: int,
        fixed_point_threshold: float,
        seed: int = 1,
    ) -> "ELM":
        """
        Build a Llama instance by initializing and loading a pre-trained model.

        Args:
            hf_model_path (str): Path to the directory containing the base LLM checkpoint files.
            lora_model_path (str): Path to the directory containing LoRA checkpoint files to finetuning base LLM for downstream tasks.
            model_args_path (str): Path to the params of ELM.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            fixed_point_solver: support 3 solvers for fixed-point: [simple, broyden, anderson].
            fixed_point_max_iters: Maximum iterations for fixed-point solving.
            fixed_point_threshold: terminate the iteration when the norm of errors below the threshold

        Returns:
            ELM: An instance of the ELM class with the loaded model and tokenizer.

        """
        assert 1 <= max_seq_len <= 8192, f"max_seq_len must be between 1 and 8192, got {max_seq_len}."
        assert os.path.isdir(hf_model_path), f"Checkpoint directory '{hf_model_path}' does not exist."

        # seed must be the same in all processes
        torch.manual_seed(seed)

        start_time = time.time()

        with open(model_args_path, "r") as f:
            params = json.loads(f.read())
        
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_path,
            torch_dtype='auto',
            low_cpu_mem_usage=True,
            device_map='cpu',
            trust_remote_code=True
        )
        generation_config = model.generation_config
        
        converted_state_dict, converted_modelargs = hf_to_meta(model.state_dict(), model.config)

        if "Llama" in hf_model_path:
            converted_modelargs.update({"model_type": "llama"})
        elif "Qwen" in hf_model_path:
            converted_modelargs.update({"model_type": "qwen"})

        model_args: ModelArgs = ModelArgs(
            fixed_point_solver=fixed_point_solver,
            fixed_point_max_iters=fixed_point_max_iters,
            fixed_point_threshold=fixed_point_threshold,
            **params,
        )

        tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(converted_state_dict, strict=False)

        if lora_model_path is not None:
            lora_checkpoint = torch.load(lora_model_path, map_location="cpu")
            state = lora_checkpoint["model"]
            # Merge LoRA weights into Transformer.
            for i in range(params["n_layers"]):
                loraq_weight = state[f"layers.{i}.attention.lora_q.linear_A.weight"].T @ state[f"layers.{i}.attention.lora_q.linear_B.weight"].T
                lorak_weight = state[f"layers.{i}.attention.lora_k.linear_A.weight"].T @ state[f"layers.{i}.attention.lora_k.linear_B.weight"].T
                lorav_weight = state[f"layers.{i}.attention.lora_v.linear_A.weight"].T @ state[f"layers.{i}.attention.lora_v.linear_B.weight"].T
                lorao_weight = state[f"layers.{i}.attention.lora_o.linear_A.weight"].T @ state[f"layers.{i}.attention.lora_o.linear_B.weight"].T
                lora1_weight = state[f"layers.{i}.feed_forward.lora_1.linear_A.weight"].T @ state[f"layers.{i}.feed_forward.lora_1.linear_B.weight"].T
                lora2_weight = state[f"layers.{i}.feed_forward.lora_2.linear_A.weight"].T @ state[f"layers.{i}.feed_forward.lora_2.linear_B.weight"].T
                lora3_weight = state[f"layers.{i}.feed_forward.lora_3.linear_A.weight"].T @ state[f"layers.{i}.feed_forward.lora_3.linear_B.weight"].T

                device = model.layers[i].attention.wq.weight.device

                model.layers[i].attention.wq.weight.data += loraq_weight.data.T.to(device)
                model.layers[i].attention.wk.weight.data += lorak_weight.data.T.to(device)
                model.layers[i].attention.wv.weight.data += lorav_weight.data.T.to(device)
                model.layers[i].attention.wo.weight.data += lorao_weight.data.T.to(device)
                model.layers[i].feed_forward.w1.weight.data += lora1_weight.data.T.to(device)
                model.layers[i].feed_forward.w2.weight.data += lora2_weight.data.T.to(device)
                model.layers[i].feed_forward.w3.weight.data += lora3_weight.data.T.to(device)

        elm_checkpoint = torch.load(elm_ckpt_path, map_location="cpu")
        state = elm_checkpoint["model"]
        model.load_state_dict(state, strict=False)

        print(f"Loaded and merged in {time.time() - start_time:.2f} seconds")

        return ELM(model, tokenizer, generation_config)

    def __init__(self, model: Transformer, tokenizer: AutoTokenizer, generation_config):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        logsteps: bool = True,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            logsteps (bool, optional): Flag indicating whether to log iteration steps of fixed-point solving. Defaults to True.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]], Optional[List[List[float]]]]: 
                A tuple containing generated token sequences and, 
                if logprobs is True, corresponding token log probabilities; 
                if logsteps is True, corresponding token iteration steps.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.

        """
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [t[:params.max_seq_len] for t in prompt_tokens]
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_token_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)
        if logsteps:
            token_logsteps = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits, steps = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )
            token_logsteps = steps

        stop_tokens = torch.tensor(list(self.generation_config.eos_token_id))

        for cur_pos in range(min_prompt_len, total_len):
            logits, steps = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            
            if logsteps:
                token_logsteps[:, prev_pos + 1 : cur_pos + 1] = steps

            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                torch.isin(next_token, stop_tokens)
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        if logsteps:
            token_logsteps = token_logsteps.tolist()
        out_tokens, out_logprobs, out_logsteps = [], [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            steps = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            if logsteps:
                steps = token_logsteps[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            for stop_token in self.generation_config.eos_token_id:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                    probs = probs[:eos_idx] if logprobs else None
                    steps = steps[:eos_idx] if logsteps else None
                except ValueError:
                    pass
            out_tokens.append(toks)
            out_logprobs.append(probs)
            out_logsteps.append(steps)
        return (out_tokens, out_logprobs if logprobs else None, out_logsteps if logsteps else None)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        logsteps: bool = True,
        echo: bool = False,
    ) -> List:
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            logsteps (bool, optional): Flag indicating whether to log iteration steps of fixed-point solving. Defaults to True.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            List: List of dicts, each containing the generated text completion and related info.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        # prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        prompts_ = []
        for x in prompts:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": x}
            ]
            prompts_.append(self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            ))
        prompt_tokens = self.tokenizer(prompts_, add_special_tokens=False)['input_ids']
        generation_tokens, generation_logprobs, generation_steps = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            logsteps=logsteps,
            echo=echo,
        )

        return [
            {
                "generation": self.tokenizer.decode(generation_tokens[i], skip_special_tokens=True),
                "prompt_tokens": prompt_tokens,
                "tokens": [self.tokenizer.decode(x) for x in generation_tokens[i]],
                "logprobs": generation_logprobs[i] if logprobs else None,
                "logsteps": generation_steps[i] if logsteps else None,
            }
            for i in range(len(generation_tokens))
        ]


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
