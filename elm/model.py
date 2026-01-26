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

    prune_interval_start_layer: int = 1
    prune_interval_len: int = 8
    
    fixed_point_solver: str = "simple"
    fixed_point_max_iters: int = 8
    fixed_point_threshold: float = 0.0
    fixed_point_one_step_kv: bool = True


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
        
        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        causal_mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            causal mask (torch.Tensor, optional): Attention causal mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos: start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos: start_pos + seqlen] = xv
        
        keys = self.cache_k[:bsz, :start_pos + seqlen]
        values = self.cache_v[:bsz, :start_pos + seqlen]

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
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
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

    def forward(self, x):
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

        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
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

        """
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        
        return out


class FixedPointAttention(nn.Module):
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
        
        self.fixed_point_one_step_kv = args.fixed_point_one_step_kv
        if args.fixed_point_one_step_kv:
            iters_cache = 1
        else:
            iters_cache = args.fixed_point_max_iters
            
        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                iters_cache,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                iters_cache,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
        self,
        fixed_point_iter: int,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        causal_mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            causal mask (torch.Tensor, optional): Attention causal mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        if self.fixed_point_one_step_kv:
            iter_id = 0
        else:
            iter_id = fixed_point_iter
            
        self.cache_k[:bsz, iter_id, start_pos: start_pos + seqlen] = xk
        self.cache_v[:bsz, iter_id, start_pos: start_pos + seqlen] = xv
        
        keys = self.cache_k[:bsz, iter_id, :start_pos + seqlen]
        values = self.cache_v[:bsz, iter_id, :start_pos + seqlen]

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
        return self.wo(output)
    

class FixedPointLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = FixedPointAttention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            intermediate_size=args.intermediate_size
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.fixed_point_solver = args.fixed_point_solver
        self.fixed_point_max_iters = args.fixed_point_max_iters
        self.fixed_point_threshold = args.fixed_point_threshold
        self.fixed_point_one_step_kv = args.fixed_point_one_step_kv

        self.proj = nn.Linear(args.dim * 2, args.dim)

    def forward_step(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        num_iter: int,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        z = torch.cat([z, x], dim=-1)
        z = self.proj(z)
        z = z + self.attention(num_iter, self.attention_norm(z), start_pos, freqs_cis, mask)
        z = z + self.feed_forward(self.ffn_norm(z))
        return z

    def forward_simple(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape

        z = torch.zeros_like(x)
        for i in range(self.fixed_point_max_iters):
            h = self.forward_step(z, x, i, start_pos, freqs_cis, mask)
            if self.fixed_point_one_step_kv:
                if seqlen == 1 and torch.linalg.vector_norm(h - z, ord=2, dim=-1, keepdim=True) < self.fixed_point_threshold:
                    z = h
                    break
            z = h
        steps = torch.full((bsz, seqlen), i + 1).type_as(x)

        return z, steps

    def forward_anderson(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape

        z = torch.zeros_like(x)
        
        lam = 1e-4
        tau = 1.0
        k = 5 # history length
        f = lambda z, n: self.forward_step(z, x, n, start_pos, freqs_cis, mask)

        # Initialize lists to store past values and their images under the fixed-point function
        X = []
        F = []

        # Initialize the first two values for X and F
        X += [z]
        F += [f(z, 0)]
        X += [F[0]]
        F += [f(F[0], 1)]
        
        for i in range(2, self.fixed_point_max_iters):
            n = min(k, i)

            X_ = torch.stack(X[i - n: i], dim=2)
            F_ = torch.stack(F[i - n: i], dim=2)
            G = F_ - X_

            H = torch.zeros(bsz, seqlen, n + 1, n + 1).type_as(z)
            H[:, :, 0, 1:] = H[:, :, 1:, 0] = 1
            y = torch.zeros(bsz, seqlen, n + 1,     1).type_as(z)
            y[:, :, 0] = 1
            H[:, :, 1: n + 1, 1: n + 1] = G @ G.transpose(2, 3) + lam * torch.eye(n).type_as(z)[None, None]
            
            try:
                alpha = torch.linalg.solve(H.float(), y.float())[:, :, 1: n + 1, 0]
            except RuntimeError as e:
                if "singular" in str(e):
                    print("Matrix is singular, falling back to lstsq...")
                    alpha = torch.linalg.lstsq(H.float(), y.float()).solution[:, :, 1: n + 1, 0]
                else:
                    raise e
            
            alpha = alpha.type_as(z)

            X += [tau * (alpha[:, :, None] @ F_)[:, :, 0] + (1 - tau) * (alpha[:, :, None] @ X_)[:, :, 0]]
            F += [f(X[i], i)]
            # print(f"iter {i} norm = {torch.norm(X[i] - F[i])}")
            
            if seqlen == 1 and torch.linalg.vector_norm(F[-1] - X[-1], ord=2, dim=-1, keepdim=True) < self.fixed_point_threshold:
                break
        
        steps = torch.full((bsz, seqlen), i + 1).type_as(x)

        return F[-1], steps

    def forward_broyden(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        def step(update, x0, g0, g):
            '''
            ``update`` is the propsoed direction of update.
            '''
            s = 1.0

            x_est = x0 + s * update
            g0_new = g(x_est)
            return x_est, g0_new, x_est - x0, g0_new - g0

        def rmatvec(part_Us, part_VTs, x):
            # Compute x^T(-I + UV^T)
            # x: (N, D)
            # part_Us: (N, D, L_thres)
            # part_VTs: (N, L_thres, D)
            if part_Us.nelement() == 0:
                return -x
            xTU = torch.einsum('bsd, bsdl -> bsl', x, part_Us)             # (B, L_thres)
            return -x + torch.einsum('bsl, bsld -> bsd', xTU, part_VTs)    # (B, D)

        def matvec(part_Us, part_VTs, x):
            # Compute (-I + UV^T)x
            # x: (B, D)
            # part_Us: (B, D, L_thres)
            # part_VTs: (B, L_thres, D)
            if part_Us.nelement() == 0:
                return -x
            VTx = torch.einsum('bsld, bsd -> bsl', part_VTs, x)            # (B, L_thres)
            return -x + torch.einsum('bsdl, bsl -> bsd', part_Us, VTx)     # (B, D)

        bsz, seqlen, dim = x.shape
        z = torch.zeros_like(x)
        max_iter = self.fixed_point_max_iters

        g = lambda z: self.forward_step(z, x, 0, start_pos, freqs_cis, mask) - z

        gz = g(z)
        nstep = 0

        Us = torch.zeros(bsz, seqlen, dim, max_iter).type_as(z)   # One can also use an L-BFGS scheme to further reduce memory
        VTs = torch.zeros(bsz, seqlen, max_iter, dim).type_as(z)
        update = -matvec(Us[:, :, :, :nstep], VTs[:, :, :nstep], gz)

        while nstep < max_iter - 1:
            # We omit line search here to controll the number of iters.
            z, gz, delta_z, delta_gz = step(update, z, gz, g)
            nstep += 1

            if seqlen == 1 and torch.linalg.vector_norm(gz, ord=2, dim=-1, keepdim=True) < self.fixed_point_threshold:
                break

            # Update the inverses Jacobian approximation using the Broyden's update formula
            part_Us, part_VTs = Us[:, :, :, :nstep - 1], VTs[:, :, :nstep - 1, :]
            vT = rmatvec(part_Us, part_VTs, delta_z)
            u = (delta_z - matvec(part_Us, part_VTs, delta_gz)) / torch.einsum('bsd,bsd->bs', vT, delta_gz)[:, :, None]
            vT[vT != vT] = 0
            u[u != u] = 0
            VTs[:, :, (nstep - 1) % max_iter] = vT
            Us[:, :, :, (nstep - 1) % max_iter] = u
            update = -matvec(Us[:, :, :, :nstep], VTs[:, :, :nstep], gz)

        steps = torch.full((bsz, seqlen), nstep + 1).type_as(z)

        return z, steps

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        if self.fixed_point_solver == "simple":
            forward = self.forward_simple
        elif self.fixed_point_solver == "anderson":
            forward = self.forward_anderson
        elif self.fixed_point_solver == "broyden":
            forward = self.forward_broyden
        else:
            raise ValueError("Unsupported solver for fixed-point!")

        return forward(x, start_pos, freqs_cis, mask)


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            prune_interval_start_layer (int): the first pruned layer index.
            prune_interval_len (int): the length of pruned layer interval.
            prune_interval_replace_module (nn.Module): the module to replace the pruned layer interval.

        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = Embedding(params.vocab_size, params.dim)

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
        self.prune_interval_replace_module = FixedPointLayer(params)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        Perform a forward pass through the ELM.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            output (torch.Tensor): Output logits after applying the ELM.
            steps (torch.Tensor): Iteration steps of FixedPointLayer in ELM.

        """
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                mask
            ]).type_as(h)

        prune_layer_id = list(range(self.prune_interval_start_layer, self.prune_interval_start_layer + self.prune_interval_len))
        for i, layer in enumerate(self.layers):
            if i == prune_layer_id[0]:
                h, steps = self.prune_interval_replace_module.forward(h, start_pos, freqs_cis, mask)
            elif i in prune_layer_id:
                continue
            else:
                h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)
        output = self.output(h).float()
        
        return output, steps