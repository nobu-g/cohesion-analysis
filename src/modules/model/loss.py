import torch
from torch import nn

eps = 1e-6


def cross_entropy_loss(
    output: torch.Tensor,  # (b, rel, seq, seq)
    target: torch.Tensor,  # (b, rel, seq, seq)
    mask: torch.Tensor,  # (b, rel, seq, seq)
) -> torch.Tensor:  # ()
    log_softmax = torch.log_softmax(output, dim=3)  # (b, rel, seq, seq)
    # reduce using masked mean (target ⊆ mask)
    # TODO: 最後の次元についてまず mean を取る（cross entropy の定義）
    return torch.sum(-log_softmax * target * mask).div(torch.sum(target * mask) + eps)


def binary_cross_entropy_with_logits(
    output: torch.Tensor,  # (b, seq, seq)
    target: torch.Tensor,  # (b, seq, seq)
    mask: torch.Tensor,  # (b, seq, seq)
) -> torch.Tensor:  # ()
    losses = nn.functional.binary_cross_entropy_with_logits(output, target.float(), reduction="none")  # (b, seq, seq)
    # reduce using masked mean
    return torch.sum(losses * mask).div(torch.sum(mask) + eps)
