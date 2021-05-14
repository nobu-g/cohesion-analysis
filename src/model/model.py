from typing import Tuple

import torch
import torch.nn as nn
from transformers import BertModel

from .sub.mask import get_mask
from .loss import cross_entropy_pas_loss


class CAModel(nn.Module):
    def __init__(self,
                 bert_model: str,
                 vocab_size: int,
                 dropout: float,
                 num_case: int,
                 coreference: bool,
                 **_
                 ) -> None:
        super().__init__()

        self.bert: BertModel = BertModel.from_pretrained(bert_model)
        self.bert.resize_token_embeddings(vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.num_case = num_case + int(coreference)
        bert_hidden_size = self.bert.config.hidden_size

        self.l_prd = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.l_arg = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.out = nn.Linear(bert_hidden_size, 1, bias=False)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                **_
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        mask = get_mask(attention_mask, ng_token_mask)
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)

        h_p = self.l_prd(self.dropout(sequence_output))  # (b, seq, case*hid)
        h_a = self.l_arg(self.dropout(sequence_output))  # (b, seq, case*hid)
        h_p = h_p.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
        h_a = h_a.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
        h_pa = torch.tanh(self.dropout(h_p.unsqueeze(2) + h_a.unsqueeze(1)))  # (b, seq, seq, case, hid)
        # -> (b, seq, seq, case, 1) -> (b, seq, seq, case) -> (b, seq, case, seq)
        output = self.out(h_pa).squeeze(-1).transpose(2, 3).contiguous()
        output += (~mask).float() * -1024.0

        loss = cross_entropy_pas_loss(output, target)

        return loss, output


class CorefCAModel(nn.Module):
    def __init__(self,
                 bert_model: str,
                 vocab_size: int,
                 dropout: float,
                 num_case: int,
                 coreference: bool,
                 ) -> None:
        super().__init__()

        self.bert: BertModel = BertModel.from_pretrained(bert_model)
        self.bert.resize_token_embeddings(vocab_size)
        self.dropout = nn.Dropout(dropout)

        assert coreference is True
        self.num_case = num_case + int(coreference)
        bert_hidden_size = self.bert.config.hidden_size

        self.l_context = nn.Linear(bert_hidden_size, bert_hidden_size)

        self.l_prd = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.l_arg = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.out = nn.Linear(bert_hidden_size, 1, bias=False)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                progress: float = 1.0,         # learning progress (0 ~ 1)
                **_
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        batch_size, seq_len = input_ids.size()
        mask = get_mask(attention_mask, ng_token_mask)  # (b, seq, case, seq)
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
        # -> (b, seq, case*hid) -> (b, seq, case, hid)
        h_p = self.l_prd(self.dropout(sequence_output)).view(batch_size, seq_len, self.num_case, -1)
        # -> (b, seq, case*hid) -> (b, seq, case, hid)
        h_a = self.l_arg(self.dropout(sequence_output)).view(batch_size, seq_len, self.num_case, -1)

        gold_ratio = 0.5 - progress * 0.5 if self.training else 0
        gold_mask = torch.rand_like(input_ids, dtype=torch.float).lt(gold_ratio).unsqueeze(-1)  # (b, seq, 1)

        # assuming coreference is the last case
        # (b, seq, seq, hid)
        h_coref = torch.tanh(self.dropout(h_p[:, :, -1, :].unsqueeze(2) + h_a[:, :, -1, :].unsqueeze(1)))
        out_coref = self.out(h_coref).squeeze(-1)  # (b, seq, seq)
        out_coref += (~mask[:, :, -1, :]).float() * -1024.0
        # (b, seq, seq)
        annealed_out_coref = (~target[:, :, -1, :] * -1024.0) * gold_mask + out_coref.detach() * ~gold_mask

        hid_context = self.l_context(self.dropout(sequence_output))  # (b, seq, hid)
        hid_context = hid_context.unsqueeze(2).expand(batch_size, seq_len, self.num_case, -1)  # (b, seq, case, hid)
        context = torch.einsum('bjch,bij->bich', hid_context, annealed_out_coref.softmax(dim=2))  # (b, seq, case, hid)
        h_a += context

        h_pa = torch.tanh(self.dropout(h_p.unsqueeze(2) + h_a.unsqueeze(1)))  # (b, seq, seq, case, hid)

        # -> (b, seq, seq, case, 1) -> (b, seq, seq, case) -> (b, seq, case, seq)
        output = self.out(h_pa).squeeze(-1).transpose(2, 3).contiguous()
        output += (~mask).float() * -1024.0

        loss = cross_entropy_pas_loss(output, target)

        return loss, output
