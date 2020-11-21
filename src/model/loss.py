import torch


def cross_entropy_pas_loss(output: torch.Tensor,  # (b, seq, case, seq)
                           target: torch.Tensor,  # (b, seq, case, seq)
                           ) -> torch.Tensor:     # ()
    log_softmax = torch.log_softmax(output, dim=3)  # (b, seq, case, seq)
    eps = 1e-6
    return torch.sum(-log_softmax * target) / (torch.sum(target) + eps)
