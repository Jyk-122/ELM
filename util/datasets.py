import os
import sys
import json
import numpy as np
import copy
from tqdm import tqdm
from typing import Optional, Tuple, Any, List, Dict

import torch
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother


IGNORE_TOKEN_ID = LabelSmoother.ignore_index

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def preprocess_belle(
    raw_data,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant."
) -> Dict:
    sources = [example["conversations"] for example in raw_data]
    
    roles = {"human": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])

    input_ids = torch.tensor(input_ids, dtype=torch.int64)
    targets = torch.tensor(targets, dtype=torch.int64)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_alpaca_qwen(
    raw_data,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant."
) -> Dict:
    
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    max_length = 0
    input_ids, targets = [], []
    for data in tqdm(raw_data):
        input_id, target = [], []

        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        
        _input_id = tokenizer(roles["user"]).input_ids + nl_tokens + \
            tokenizer(data["instruction"]).input_ids + \
            tokenizer(data["input"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
        target += _target
        
        _input_id = tokenizer(roles["assistant"]).input_ids + nl_tokens + \
            tokenizer(data["output"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(roles["assistant"]).input_ids) + \
                    _input_id[len(tokenizer(roles["assistant"]).input_ids)+1:-2] + [im_end] + nl_tokens
        target += _target
        assert len(input_id) == len(target)

        max_length = max(max_length, len(input_id))

        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))

        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])

    input_ids = torch.tensor(input_ids, dtype=torch.int64)
    targets = torch.tensor(targets, dtype=torch.int64)
    input_ids_mask = torch.where(input_ids == tokenizer.pad_token_id, 0, 1)
    targets_mask = torch.where(targets == IGNORE_TOKEN_ID, 0, 1)

    print(f"############### MAX_LENGTH={max_length} #################")

    return dict(
        input_ids=input_ids,
        labels=targets,
        input_ids_mask=input_ids_mask,
        labels_mask=targets_mask,
    )


def preprocess_alpaca_llama(
    raw_data,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant."
) -> Dict:
    
    roles = {"user": "user", "assistant": "assistant"}

    im_start = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    im_end = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    header_start = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    header_end = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    nl_tokens = tokenizer('\n\n', add_special_tokens=False).input_ids
    _system = tokenizer('system', add_special_tokens=False).input_ids
    _user = tokenizer('user', add_special_tokens=False).input_ids
    _assistant = tokenizer('assistant', add_special_tokens=False).input_ids
    _system_message = tokenizer(system_message, add_special_tokens=False).input_ids

    max_length = 0
    input_ids, targets = [], []
    for data in tqdm(raw_data):
        input_id, target = [], []

        system = [im_start] + [header_start] + _system + [header_end] + nl_tokens + _system_message + [im_end]
        input_id += system
        target += [im_start] + [header_start] + [IGNORE_TOKEN_ID] * len(_system) + [header_end] + [IGNORE_TOKEN_ID] * (len(nl_tokens) + len(_system_message)) + [im_end]
        assert len(input_id) == len(target)
        
        _input_id = [header_start] + _user + [header_end] + nl_tokens + \
            tokenizer(data["instruction"] + data["input"], add_special_tokens=False).input_ids + [im_end]
        input_id += _input_id
        _target = [header_start] + [IGNORE_TOKEN_ID] * len(_user) + [header_end] + [IGNORE_TOKEN_ID] * len(nl_tokens) + \
            [IGNORE_TOKEN_ID] * (len(tokenizer(data["instruction"] + data["input"], add_special_tokens=False).input_ids)) + [im_end]
        target += _target
        assert len(input_id) == len(target)
        
        _input_id = [header_start] + _assistant + [header_end] + nl_tokens + \
            tokenizer(data["output"]).input_ids + [im_end]
        input_id += _input_id
        _target = [header_start] + [IGNORE_TOKEN_ID] * len(_assistant) + [header_end] + [IGNORE_TOKEN_ID] * len(nl_tokens) + \
            tokenizer(data["output"]).input_ids + [im_end]
        target += _target
        assert len(input_id) == len(target)

        max_length = max(max_length, len(input_id))

        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))

        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])

    input_ids = torch.tensor(input_ids, dtype=torch.int64)
    targets = torch.tensor(targets, dtype=torch.int64)
    input_ids_mask = torch.where(input_ids == tokenizer.pad_token_id, 0, 1)
    targets_mask = torch.where(targets == IGNORE_TOKEN_ID, 0, 1)

    print(f"############### MAX_LENGTH={max_length} #################")

    return dict(
        input_ids=input_ids,
        labels=targets,
        input_ids_mask=input_ids_mask,
        labels_mask=targets_mask,
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, dataset_name, dataset_path, partition, data_type, prompt_type, tokenizer_path, max_len, unittest):
        super(SupervisedDataset, self).__init__()

        dataset_name = dataset_name.lower()
        dataset_path = os.path.join(dataset_path, dataset_name)

        if partition == "train":
            data_path = os.path.join(dataset_path, "train.json")
        elif partition == "test":
            data_path = os.path.join(dataset_path, "test.json")
        else:
            raise ValueError(f"Invalid partiton type: {partition}")

        with open(data_path, "r") as f:
            raw_data = json.load(f)
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

        if data_type == "instruction" and prompt_type == "llama":
            preprocess = preprocess_alpaca_llama
        elif data_type == "instruction" and prompt_type == "qwen":
            preprocess = preprocess_alpaca_qwen
        elif data_type == "chat" and prompt_type == "qwen":
            preprocess = preprocess_belle
        else:
            raise ValueError(f"Invalid partiton type: {partition}")

        if unittest <= 0:
            data_dict = preprocess(raw_data, tokenizer, max_len)
        else:
            data_dict = preprocess(raw_data[:unittest], tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.input_ids_mask = data_dict["input_ids_mask"]
        self.labels_mask = data_dict["labels_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.labels[index], self.input_ids_mask[index], self.labels_mask[index]
