import math
import torch
from torch import nn
from .basic_blocks import Embedding, Linear

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float=1e-5,
        device: torch.device | None=None,
        dtype: torch.dtype | None=None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        input:
            (batch_size, sequence_length, d_model)
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.weight

        return result.to(in_dtype)


class PositionWiseFeedForward(nn.Module):
    def __init__(
        self,
        d_in: int,
        device: torch.device | None=None,
        dtype: torch.dtype | None=None
    ):
        super().__init__()
        self.d_in = d_in
        d_ff = int(8 * d_in / 3)
        d_ff -= d_ff % 64
        self.d_ff = d_ff

        self.w1 = Linear(d_in, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_in, device=device, dtype=dtype)
        self.w3 = Linear(d_in, d_ff, device=device, dtype=dtype)

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        """ x * sigmoid(x) """
        return torch.sigmoid(x) * x

    def swiglu(self, x: torch.Tensor, w1: nn.Module, w3: nn.Module) -> torch.Tensor:
        """ SiLU(W_1x) * W_3x"""
        return self.silu(w1(x)) * w3(x)

    def forward(self, x):
        """ SwiGLU PointWise Feed Forward """
        return self.w2(self.swiglu(x, self.w1, self.w3))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None=None,
    ):
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even for RoPE, but got {d_k}")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # shape: (d_k // 2,)
        pair_idx = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (pair_idx / d_k))

        # shape: (max_seq_len,)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        # shape: (max_seq_len, d_k//2)
        # angles[i, k]:
        angles = torch.outer(positions, inv_freq)

        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        input:
            x: (..., seq_len, d_k)
            token_positions: (..., seq_len)
        output:
            (..., seq_len, d_k)
        """
        if x.shape[-1] != self.d_k:
            raise ValueError(
                f"Expected x.shape[-1] == {self.d_k}, but got {x.shape[-1]}"
            )

        seq_len = x.shape[-2]
        if token_positions.shape[-1] != seq_len:
            raise ValueError(
                f"token_positions.shape[-1] must equal seq_len={seq_len}, "
                f"but got {token_positions.shape[-1]}"
            )

        if torch.any(token_positions < 0) or torch.any(token_positions >= self.max_seq_len):
            raise ValueError("token_positions contains indices out of range")

        # 让 token_positions 可以广播到 x.shape[:-1]
        target_shape = x.shape[:-1]
        token_positions = torch.broadcast_to(token_positions, target_shape)

        cos = self.cos_cached[token_positions].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[token_positions].to(dtype=x.dtype, device=x.device)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out

def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    x_max, _ = x.max(dim=i, keepdim=True)
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=i, keepdim=True)

def masked_softmax(
    x: torch.Tensor,
    mask: torch.Tensor,
    i: int,
    inf: int=1e6
) -> torch.Tensor:
    if mask.dtype != torch.bool:
        raise ValueError("mask must be a boolean tensor")

    extra_dims = x.dim() - mask.dim()
    mask = mask.view((1,) * extra_dims + mask.shape)

    x = x.masked_fill(~mask, -inf)
    return softmax(x, i)

def scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor | None=None,
):
    d_k = queries.size(-1)

    scores = torch.matmul(queries ,keys.transpose(-1, -2))
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        attn = masked_softmax(scores, mask, -1)
    else:
        attn = softmax(scores, -1)

    return torch.matmul(attn, values)

class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        theta: float = 10000.0,
        max_seq_len: int = 4096
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = Linear(d_model, d_model, device=device)
        self.w_k = Linear(d_model, d_model, device=device)
        self.w_v = Linear(d_model, d_model, device=device)

        self.w_o = Linear(d_model, d_model, device=device)

        self.rope_q = RotaryPositionalEmbedding(
            theta=theta,
            d_k=self.d_k,
            max_seq_len=max_seq_len,
            device=device,
        )

        self.rope_k = RotaryPositionalEmbedding(
            theta=theta,
            d_k=self.d_k,
            max_seq_len=max_seq_len,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input:
            x: (..., seq_len, d_model)
        output:
            (..., seq_len, d_model)
        """
        seq_len = x.shape[-2]

        # 映射成 queries, keys, values
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # 切分成多个头
        q = q.view(*x.shape[:-2], seq_len, self.num_heads, self.d_k)
        k = k.view(*x.shape[:-2], seq_len, self.num_heads, self.d_k)
        v = v.view(*x.shape[:-2], seq_len, self.num_heads, self.d_k)

        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)

        token_position = torch.arange(seq_len, device=x.device)

        q = self.rope_q(q, token_position)
        k = self.rope_k(k, token_position)

        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )

        attn = scaled_dot_product_attention(q, k, v, mask)
        
        attn = attn.transpose(-3, -2)
        attn = attn.reshape(*x.shape[-2], seq_len, self.d_model)

        return self.w_o(attn)