from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np

from utils.parse_config import ConfigParser
from utils import prepare_device


class Inference:
    def __init__(self,
                 config: ConfigParser,
                 model: nn.Module,
                 precision_threshold: float = 0.0,
                 recall_threshold: float = 0.0,
                 logger=None):
        self.config = config
        self.logger = logger if logger else config.get_logger('inference')
        self.p_threshold: float = precision_threshold
        self.r_threshold: float = recall_threshold

        self.device, self.device_ids = prepare_device(config['n_gpu'], self.logger)
        self.state_dicts = []
        checkpoints = [config.resume] if config.resume is not None else list(config.save_dir.glob('**/model_best.pth'))
        for checkpoint in checkpoints:
            self.logger.info(f'Loading checkpoint: {checkpoint} ...')
            state_dict = torch.load(checkpoint, map_location=self.device)['state_dict']
            self.state_dicts.append({k.replace('module.', ''): v for k, v in state_dict.items()})

        self.model = model

    def __call__(self, data_loader) -> tuple:
        total_output: Optional[Tuple[np.ndarray]] = None
        total_loss = 0.0
        for state_dict in self.state_dicts:
            model = self._prepare_model(state_dict)
            loss, *output = self._forward(model, data_loader)
            total_output = tuple(t + o for t, o in zip(total_output, output)) if total_output is not None else output
            total_loss += loss
        avg_loss = total_loss / len(self.state_dicts)

        predictions = []
        for i, output in enumerate(total_output):
            if len(output.shape) == 4:
                output = Inference._softmax(output, axis=3)
                null = data_loader.dataset.special_to_index['NULL']
                if data_loader.dataset.coreference:
                    output[:, :, :-1, null] += (output[:, :, :-1] < self.p_threshold).all(axis=3).astype(np.int) * 1024
                    output[:, :, :-1, null] -= (output[:, :, :-1] < self.r_threshold).all(axis=3).astype(np.int) * 1024
                    na = data_loader.dataset.special_to_index['NA']
                    output[:, :, -1, na] += (output[:, :, -1] < self.p_threshold).all(axis=2).astype(np.int) * 1024
                    output[:, :, -1, na] -= (output[:, :, -1] < self.r_threshold).all(axis=2).astype(np.int) * 1024
                else:
                    output[:, :, :, null] += (output < self.p_threshold).all(axis=3).astype(np.int) * 1024
                    output[:, :, :, null] -= (output < self.r_threshold).all(axis=3).astype(np.int) * 1024
                predictions.append(np.argmax(output, axis=3))
            elif len(output.shape) == 1:
                predictions.append((output > 0.5).astype(np.int))
            else:
                raise ValueError(f'unexpected output shape: {output.shape}')

        return avg_loss, *predictions

    def _prepare_model(self, state_dict: dict):
        self.model.load_state_dict(state_dict)
        self.model.eval()
        model = self.model.to(self.device)
        if len(self.device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.device_ids)
        return model

    def _forward(self, model, data_loader) -> Tuple[float, ...]:
        total_loss = 0.0
        outputs: List[Tuple[np.ndarray, ...]] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                batch = {label: t.to(self.device, non_blocking=True) for label, t in batch.items()}

                loss, *output = model(**batch)

                if len(loss.size()) > 0:
                    loss = loss.mean()
                outputs.append(tuple(o.cpu().numpy() for o in output))
                total_loss += loss.item() * output[0].size(0)
        avg_loss: float = total_loss / len(data_loader.dataset)
        return avg_loss, *(np.concatenate(outs, axis=0) for outs in zip(*outputs))

    @staticmethod
    def _softmax(x: np.ndarray, axis: int):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / (e_x.sum(axis=axis, keepdims=True) + 1e-8)
