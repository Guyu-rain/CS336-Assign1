import torch
import math
from torch import nn

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None=None,
        dtype: torch.dtype | None=None,
    ):
        super().__init__()

        self.in_features = in_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=dtype, device=device))
        std = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: batch_size, seq_len, in_sequences """
        return x @ self.weight.transpose(1, 0)

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None=None,
        dtype: torch.dtype | None=None
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        std = math.sqrt(2.0 / (num_embeddings + embedding_dim))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor :
        """
        input:
            token_ids: batch_size, sequence_length
        output：
            batch_size, sequence_length, embedding_dim
        """
        return self.weight[token_ids]

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