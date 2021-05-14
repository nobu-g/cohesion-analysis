from typing import List, Tuple, Dict

import torch
import numpy as np
from data_loader.dataset.pas_dataset import PASDataset
from torch.utils.data import DataLoader


class PASDataLoader(DataLoader):
    def __init__(self,
                 dataset: PASDataset,
                 batch_size: int,
                 shuffle: bool,
                 num_workers: int,
                 pin_memory: bool,
                 ):
        init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'collate_fn': broadcast_collate_fn,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
        }
        super().__init__(**init_kwargs)


def broadcast_collate_fn(batch: List[Tuple[np.ndarray, ...]]) -> Dict[str, torch.Tensor]:
    input_ids, attention_mask, segment_ids, ng_token_mask, target, deps, task, overt_mask = zip(*batch)  # Tuple[list]
    ng_token_mask = np.broadcast_arrays(*ng_token_mask)
    target = np.broadcast_arrays(*target)
    deps = np.broadcast_arrays(*deps)
    inputs = (input_ids, attention_mask, segment_ids, ng_token_mask, target, deps, task, overt_mask)
    labels = ('input_ids', 'attention_mask', 'segment_ids', 'ng_token_mask', 'target', 'deps', 'task', 'overt_mask')
    return {label: torch.as_tensor(np.stack(elem, axis=0)) for label, elem in zip(labels, inputs)}
