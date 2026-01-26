# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from contextlib import nullcontext
import random

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Embedding, Linear
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

@dataclass
class ModelArgs:
    model_type: str = "llama"
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    intermediate_size: Optional[int] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048

    lora_rank: int = 16
    prune_interval_start_layer: int = 1
    prune_interval_len: int = 8


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class LoRAModule(nn.Module):
    def __init__(self, in_dim : int, rank : int, out_dim : int):
        """
        Initialize the LoRA module.

        Args:
            in_dim (int): Dim of input feature.
            rank (int): Dim of hidden layer.
            out_dim (int): Dim of output tensor.

        Attributes:
            linear_A (Linear): Linear transformation A for rank reduction.
            linear_B (Linear): Linear transformation B for rank up.
            dropout (nn.Dropout): Droupout layer.

        """
        super().__init__()
        self.linear_A = nn.Linear(in_dim, rank, bias=False)
        self.linear_B = nn.Linear(rank, out_dim, bias=False)
        self.dropout = nn.Dropout(0.05)

        nn.init.normal_(self.linear_A.weight, std=1 / rank)
        nn.init.zeros_(self.linear_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_B(self.linear_A(x.float()))
        return self.dropout(x).bfloat16()
    

class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (Linear): Linear transformation for queries.
            wk (Linear): Linear transformation for keys.
            wv (Linear): Linear transformation for values.
            wo (Linear): Linear transformation for output.

        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        
        if args.model_type == "llama":
            bias = False
        elif args.model_type == "qwen":
            bias = True

        self.wq = Linear(args.dim, args.n_heads * self.head_dim, bias=bias)
        self.wk = Linear(args.dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.wv = Linear(args.dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.wo = Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        self.lora_q = LoRAModule(in_dim=args.dim, rank=args.lora_rank, out_dim=self.head_dim * self.n_local_heads)
        self.lora_k = LoRAModule(in_dim=args.dim, rank=args.lora_rank, out_dim=self.head_dim * self.n_local_kv_heads)
        self.lora_v = LoRAModule(in_dim=args.dim, rank=args.lora_rank, out_dim=self.head_dim * self.n_local_kv_heads)
        self.lora_o = LoRAModule(in_dim=args.n_heads * self.head_dim, rank=args.lora_rank, out_dim=args.dim)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        causal_mask: Optional[torch.Tensor],
        lora_apply: bool = False,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            causal mask (torch.Tensor, optional): Attention causal mask tensor.
            lora_apply (bool): Whether to apply LoRA in this module.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape

        if lora_apply:
            xq = self.wq(x) + self.lora_q(x)
            xk = self.wk(x) + self.lora_k(x)
            xv = self.wv(x) + self.lora_v(x)
        else:
            xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = xk
        values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if causal_mask is not None:
            scores = scores + causal_mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        if lora_apply:
            output = self.wo(output) + self.lora_o(output)
        else:
            output = self.wo(output)
        
        return output


class FeedForward(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        intermediate_size: Optional[int],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (Linear): Linear transformation for the first layer.
            w2 (Linear): Linear transformation for the second layer.
            w3 (Linear): Linear transformation for the third layer.

        """
        super().__init__()
        if intermediate_size is not None:
            hidden_dim = intermediate_size
        else:
            hidden_dim = int(2 * hidden_dim / 3)
            # custom dim factor multiplier
            if ffn_dim_multiplier is not None:
                hidden_dim = int(ffn_dim_multiplier * hidden_dim)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = Linear(dim, hidden_dim, bias=False)
        self.w2 = Linear(hidden_dim, dim, bias=False)
        self.w3 = Linear(dim, hidden_dim, bias=False)
        
        self.lora_1 = LoRAModule(dim, args.lora_rank, hidden_dim)
        self.lora_2 = LoRAModule(hidden_dim, args.lora_rank, dim)
        self.lora_3 = LoRAModule(dim, args.lora_rank, hidden_dim)

    def forward(self, x, lora_apply: bool = False):
        """
        Forward pass of the ffn module.

        Args:
            x (torch.Tensor): Input tensor.
            lora_apply (bool): Whether to apply LoRA in this module.

        Returns:
            torch.Tensor: Output tensor after ffn.

        """
        if lora_apply:
            w1 = self.w1(x) + self.lora_1(x)
            w3 = self.w3(x) + self.lora_3(x)
            h = F.silu(w1 * w3)
            return self.w2(h) + self.lora_2(h)
        else:
            return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.
            dynamic_start_layer (int): Indicate the layer start to apply router module.
            router (Optional[RouterModule]): Router module.

        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            args=args,
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            intermediate_size=args.intermediate_size,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        lora_apply: bool = False,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Causal masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.
            w (torch.Tensor): Output routes after applying router module.

        """
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask, lora_apply)
        out = h + self.feed_forward(self.ffn_norm(h), lora_apply)
        
        return out


class Adaptor(nn.Module):
    def __init__(self, args: ModelArgs):
        """
        Initialize an Adaptor applied before transformer layer in fixed-point module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            proj (nn.Moudle): Linear adpator.
        """
        super().__init__()
        self.proj = nn.Linear(args.dim * 2, args.dim)
        # initialize proj
        self.proj.weight.data.mul_(0.1)
        self.proj.weight.data[:, args.dim:].zero_().fill_diagonal_(1)
        self.proj.bias.data.zero_()
        
    def forward(self, z, x):
        """
        Perform the forward pass of adaptor in FixedPointLayer.

        Args:
            z (torch.Tensor): fixed-point to be solved.
            x (torch.Tensor): Conditional input tensor.
            
        Returns:
            torch.Tensor: Fused tensor after adptor.

        """
        h = torch.cat([z, x], dim=-1).float()
        return self.proj(h).type_as(x)


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        """
        Initialize an Equilibrium Language Model as Reward Model for GPPO.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            criterion (torch.nn.CrossEntropyLoss): CrossEntropy function.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (Linear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            prune_interval_start_layer (int): the first pruned layer index.
            prune_interval_len (int): the length of pruned layer interval.
            prune_interval_replace_module_adaptors (nn.Module): the adaptor modules applied before each candidate pruning layer.

        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        
        self.tok_embeddings = Embedding(params.vocab_size, params.dim)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_TOKEN_ID)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.output = Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096. 
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2, self.params.rope_theta
        )
        
        self.prune_interval_start_layer = params.prune_interval_start_layer
        self.prune_interval_len = params.prune_interval_len
        self.prune_interval_replace_module_adaptors = nn.ModuleList()
        for layer_id in range(params.n_layers - params.prune_interval_len + 1):
            self.prune_interval_replace_module_adaptors.append(Adaptor(params))

    def forward_fixed_point_layer(
        self,
        layer_id: int,
        x: torch.Tensor,
        start_pos: int,
        freq_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Perform a single forward pass step through the candidate FixedPointLayer with LoRA.

        Args:
            layer_id (int): The index of first pruned layer.
            x (torch.Tensor): Conditional input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Causal masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after one step fixed-point iteration.

        """
        z = torch.zeros_like(x)
        for _ in range(self.prune_interval_len):
            z = self.prune_interval_replace_module_adaptors[layer_id](z, x)
            z = self.layers[layer_id](z.type_as(x), start_pos, freq_cis, mask, lora_apply=True)
        
        return z
        
    def forward(
        self,
        prune_start_layer: int,
        examples: torch.Tensor,
        labels: torch.Tensor,
        example_masks: torch.Tensor,
        label_masks: torch.Tensor,
    ):
        """
        Perform a forward pass through the Equilibrium Language Model.

        Args:
            prune_start_layer (int): The selected index of first pruned layer to build ELM.
            examples (torch.Tensor): Input token indices representing whole sentences.
            labels (torch.Tensor): Input token indices representing labels.
            example_masks (torch.Tensor): Mask corresponding to sentences.
            label_masks (torch.Tensor): Mask corresponding to labels.

        Returns:
            loss_dict (Dict):
                c_loss (torch.Tensor): Output CE after applying the ELM.
                d_loss (torch.Tensor): Output distillation loss after applying the ELM.
            
        """
        bsz, seqlen = examples.shape

        h = self.tok_embeddings(examples)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seqlen]

        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(h)
        
        start_pos = 0
        
        prune_layer_id = list(range(prune_start_layer, prune_start_layer + self.prune_interval_len))
        for i in range(0, prune_layer_id[0]):
            h = self.layers[i](h, start_pos, freqs_cis, mask, lora_apply=False)
        
        h_input = h.clone()
        h_output = self.forward_fixed_point_layer(prune_start_layer, h_input, start_pos, freqs_cis, mask)
        h_output = h_output.type_as(h)
        
        for i in prune_layer_id:
            h = self.layers[i](h, start_pos, freqs_cis, mask, lora_apply=False)
            
        d_loss = F.mse_loss(h_output, h, reduction='none')
        d_loss = d_loss * example_masks[..., None]
        d_loss = d_loss.mean(dim=-1).sum() / example_masks.sum()
        
        h = h_output
        for i in range(prune_layer_id[-1] + 1, self.n_layers):
            h = self.layers[i](h, start_pos, freqs_cis, mask, lora_apply=False)

        h = self.norm(h)
        output = self.output(h)
        output = output[:, :-1, :].reshape(-1, self.vocab_size)
        labels = labels[:, 1:].flatten()
        c_loss = self.criterion(output, labels)
        
        return {'c_loss': c_loss, 'd_loss': d_loss}


class LayerWeightModule(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.weight = nn.Parameter(
            torch.zeros(
                args.n_layers - args.prune_interval_len + 1, dtype=torch.float
            )
        )
        
    def forward(self):
        return self.weight.float()


class GPPO(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.reward_model = Transformer(params)
        self.policy_model = LayerWeightModule(params)
        
    def get_action(self):
        logits = self.policy_model()
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()
    
    def forward(
        self,
        prune_start_layer: int,
        examples: torch.Tensor,
        labels: torch.Tensor,
        example_masks: torch.Tensor,
        label_masks: torch.Tensor,
    ):
        loss_dict = self.reward_model(prune_start_layer, examples, labels, example_masks, label_masks)
        rewards = -loss_dict["c_loss"]
        
        return rewards, loss_dict