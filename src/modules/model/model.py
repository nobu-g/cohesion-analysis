import math

import torch
from omegaconf import ListConfig
from torch import nn
from transformers import AutoModel, PreTrainedModel


class BaselineModel(nn.Module):
    """Naive Baseline Model"""

    def __init__(
        self,
        model_name_or_path: str,
        exophora_referents: ListConfig,
        hidden_dropout_prob: float,
        num_tasks: int,
        num_relations: int,
        **_,
    ) -> None:
        super().__init__()
        self.pretrained_model: PreTrainedModel = AutoModel.from_pretrained(model_name_or_path)
        self.pretrained_model.resize_token_embeddings(
            self.pretrained_model.config.vocab_size + len(exophora_referents) + 2,  # +2: [NULL] and [NA]
        )

        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.num_relation_types = num_relations
        hidden_size = self.pretrained_model.config.hidden_size

        self.l_source = nn.Linear(self.pretrained_model.config.hidden_size, hidden_size * self.num_relation_types)
        self.l_target = nn.Linear(self.pretrained_model.config.hidden_size, hidden_size * self.num_relation_types)
        self.out = nn.Linear(hidden_size, 1, bias=False)

        self.analysis_target_classifier = TokenBinaryClassificationHead(
            num_tasks=num_tasks,
            encoder_hidden_size=self.pretrained_model.config.hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
        )

    def forward(
        self,
        input_ids: torch.Tensor,  # (b, seq)
        attention_mask: torch.Tensor,  # (b, seq)
        token_type_ids: torch.Tensor,  # (b, seq)
        **_,
    ) -> tuple[torch.Tensor, torch.Tensor]:  # (b, rel, seq, seq), (b, task, seq)
        encoder_last_hidden_state = self.pretrained_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state  # (b, seq, hid)
        batch_size, sequence_len, hidden_size = encoder_last_hidden_state.size()

        h_src = self.l_source(self.dropout(encoder_last_hidden_state))  # (b, seq, rel*hid)
        h_tgt = self.l_target(self.dropout(encoder_last_hidden_state))  # (b, seq, rel*hid)
        h_src = h_src.view(batch_size, sequence_len, self.num_relation_types, hidden_size)  # (b, seq, rel, hid)
        h_tgt = h_tgt.view(batch_size, sequence_len, self.num_relation_types, hidden_size)  # (b, seq, rel, hid)
        h = torch.tanh(self.dropout(h_src.unsqueeze(2) + h_tgt.unsqueeze(1)))  # (b, seq, seq, rel, hid)
        # -> (b, seq, seq, rel, 1) -> (b, seq, seq, rel) -> (b, rel, seq, seq)
        relation_logits = self.out(h).squeeze(-1).permute(0, 3, 1, 2).contiguous()

        source_mask_logits = self.analysis_target_classifier(encoder_last_hidden_state)

        return relation_logits, source_mask_logits


class LoRARelationWiseWordSelectionHead(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        exophora_referents: ListConfig,
        num_relations: int,
        hidden_dropout_prob: float,
        num_tasks: int,
        rank: int = 2,
    ) -> None:
        super().__init__()
        self.pretrained_model: PreTrainedModel = AutoModel.from_pretrained(model_name_or_path)
        self.pretrained_model.resize_token_embeddings(
            self.pretrained_model.config.vocab_size + len(exophora_referents) + 2,  # +2: [NULL] and [NA]
        )
        hidden_size = self.pretrained_model.config.hidden_size
        self.l_source = nn.Linear(hidden_size, hidden_size)
        self.l_target = nn.Linear(hidden_size, hidden_size)
        self.delta_source = LoRADelta(num_relations, hidden_size, rank)
        self.delta_target = LoRADelta(num_relations, hidden_size, rank)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Parameter(torch.Tensor(hidden_size, num_relations))
        nn.init.kaiming_uniform_(self.classifier, a=math.sqrt(5))

        self.analysis_target_classifier = LoRARelationWiseTokenBinaryClassificationHead(
            num_tasks=num_tasks,
            encoder_hidden_size=self.pretrained_model.config.hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            rank=4,
        )

    def forward(
        self,
        input_ids: torch.Tensor,  # (b, seq)
        attention_mask: torch.Tensor,  # (b, seq)
        token_type_ids: torch.Tensor,  # (b, seq)
        **_,
    ) -> tuple[torch.Tensor, torch.Tensor]:  # (b, rel, seq, seq), (b, task, seq)
        encoder_last_hidden_state = self.pretrained_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state  # (b, seq, hid)
        h_source = self.l_source(self.dropout(encoder_last_hidden_state))  # (b, seq, hid)
        h_target = self.l_target(self.dropout(encoder_last_hidden_state))  # (b, seq, hid)
        delta_source_out = torch.einsum(
            "bsh,hil->bsli", encoder_last_hidden_state, self.delta_source()
        )  # (b, seq, rel, hid)
        delta_target_out = torch.einsum(
            "bsh,hil->bsli", encoder_last_hidden_state, self.delta_target()
        )  # (b, seq, rel, hid)
        source_out = h_source.unsqueeze(2) + delta_source_out  # (b, seq, rel, hid)
        target_out = h_target.unsqueeze(2) + delta_target_out  # (b, seq, rel, hid)
        # (b, seq, seq, rel, hid)
        hidden = self.dropout(self.activation(source_out.unsqueeze(2) + target_out.unsqueeze(1)))
        output = torch.einsum("bstlh,hl->bstl", hidden, self.classifier)  # (b, seq, seq, rel)
        relation_logits = output.permute(0, 3, 1, 2).contiguous()

        source_mask_logits = self.analysis_target_classifier(encoder_last_hidden_state)

        return relation_logits, source_mask_logits


class LoRADelta(nn.Module):
    def __init__(self, num_labels: int, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.dense_a = nn.Parameter(torch.Tensor(hidden_size, rank, num_labels))
        self.dense_b = nn.Parameter(torch.Tensor(rank, hidden_size, num_labels))
        nn.init.kaiming_uniform_(self.dense_a, a=math.sqrt(5))
        nn.init.zeros_(self.dense_b)

    def forward(self) -> torch.Tensor:
        return torch.einsum("hrl,ril->hil", self.dense_a, self.dense_b)  # (hid, hid, label)


class TokenBinaryClassificationHead(nn.Module):
    def __init__(self, num_tasks: int, encoder_hidden_size: int, hidden_dropout_prob: float) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.encoder_hidden_size = encoder_hidden_size
        hidden_size = self.encoder_hidden_size
        self.dense = nn.Linear(self.encoder_hidden_size, hidden_size * self.num_tasks)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(
        self,
        hidden_state: torch.Tensor,  # (b, seq, hid)
    ) -> torch.Tensor:  # (b, task, seq)
        batch_size, sequence_len, hidden_size = hidden_state.size()
        h = self.dense(self.dropout(hidden_state))  # (b, seq, task*hid)
        h = h.view(batch_size, sequence_len, self.num_tasks, hidden_size)  # (b, seq, task, hid)
        # -> (b, seq, task, 1) -> (b, seq, task) -> (b, task, seq)
        return self.classifier(torch.tanh(self.dropout(h))).squeeze(-1).permute(0, 2, 1).contiguous()


class LoRARelationWiseTokenBinaryClassificationHead(nn.Module):
    def __init__(self, num_tasks: int, encoder_hidden_size: int, hidden_dropout_prob: float, rank: int) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.encoder_hidden_size = encoder_hidden_size
        hidden_size = self.encoder_hidden_size
        self.dense = nn.Linear(self.encoder_hidden_size, hidden_size * self.num_tasks)
        self.delta = LoRADelta(num_tasks, hidden_size, rank)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Parameter(torch.Tensor(hidden_size, num_tasks))
        nn.init.kaiming_uniform_(self.classifier, a=math.sqrt(5))

    def forward(
        self,
        hidden_state: torch.Tensor,  # (b, seq, hid)
    ) -> torch.Tensor:  # (b, task, seq)
        batch_size, sequence_len, hidden_size = hidden_state.size()
        h = self.dense(self.dropout(hidden_state))  # (b, seq, task*hid)
        h = h.view(batch_size, sequence_len, self.num_tasks, hidden_size)  # (b, seq, task, hid)
        delta_out = torch.einsum("bsh,hil->bsli", hidden_state, self.delta())  # (b, seq, task, hid)
        # -> (b, seq, task, 1) -> (b, seq, task) -> (b, task, seq)
        hidden = torch.tanh(self.dropout(h + delta_out))  # (b, seq, task, hid)
        # (b, seq, task) -> (b, task, seq)
        return torch.einsum("bsth,ht->bst", hidden, self.classifier).permute(0, 2, 1).contiguous()
