from dataclasses import fields, is_dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from datamodule.dataset import CohesionDataset


def _is_included(tensor1: torch.Tensor, tensor2: torch.Tensor) -> bool:
    """Check if tensor1 is included in tensor2."""
    assert tensor1.dim() == tensor2.dim(), "The number of dimensions must be the same."
    assert tensor1.dtype == torch.bool, "The dtype of tensor1 must be bool."
    assert tensor2.dtype == torch.bool, "The dtype of tensor2 must be bool."
    return (tensor1 & tensor2).equal(tensor1)


def _dataclass_data_collator(features: list[Any]) -> dict[str, torch.Tensor]:
    first: Any = features[0]
    assert is_dataclass(first), "Data must be a dataclass"
    batch: dict[str, torch.Tensor] = {}
    for field in fields(first):
        feats = [getattr(f, field.name) for f in features]
        batch[field.name] = torch.as_tensor(feats)
    return batch


def test_target_not_masked(fixture_train_dataset: CohesionDataset):
    for batch in DataLoader(fixture_train_dataset, batch_size=2, shuffle=False, collate_fn=_dataclass_data_collator):
        assert _is_included(batch["target_label"].bool(), batch["target_mask"])
