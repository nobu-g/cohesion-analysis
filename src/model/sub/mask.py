import torch


def get_mask(attention_mask: torch.Tensor,  # (b, seq)
             ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
             ) -> torch.Tensor:             # (b, seq, case, seq)
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (b, 1, 1, seq)
    return extended_attention_mask & ng_token_mask
