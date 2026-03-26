import torch
from torch import nn
from src.model.basic_blocks import softmax

def cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    代数变换：
        logits tensor 为 `o`，正确结果是 target
        Loss = -log softmax(o)[target] = -o_target + log sum_j(e^{o_j})
    input:
        logits: (..., vocab_size)
        targets: (...)
    """
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted = logits - max_logits

    target_logits = torch.gather(
        shifted, dim=-1, index=target.unsqueeze(-1)
    ).squeeze(-1)
    logsumexp = torch.log(torch.exp(shifted).sum(dim=-1))

    loss = logsumexp - target_logits

    return loss.mean()
