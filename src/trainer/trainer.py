import re
import math
import datetime
from typing import List

import numpy as np
import torch
from sklearn.metrics import f1_score

from .base_trainer import BaseTrainer
from prediction.prediction_writer import PredictionKNPWriter
from scorer import Scorer
import data_loader.data_loaders as module_loader
from logger import TensorboardWriter


class Trainer(BaseTrainer):
    """
    Trainer class
    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, metrics, optimizer, config, train_dataset, valid_datasets, lr_scheduler=None):
        super().__init__(model, metrics, optimizer, config, train_dataset)
        self.config = config
        self.config['data_loaders']['valid']['args']['batch_size'] = self.data_loader.batch_size
        self.valid_data_loaders = {}
        for corpus, valid_dataset in valid_datasets.items():
            self.valid_data_loaders[corpus] = config.init_obj('data_loaders.valid', module_loader, valid_dataset)
        self.lr_scheduler = lr_scheduler
        self.log_step = math.ceil(len(self.data_loader.dataset) / np.sqrt(self.data_loader.batch_size) / 200)
        self.writer = TensorboardWriter(config.log_dir)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        self.optimizer.zero_grad()

        total_loss = 0
        accum_count = 0
        for step, batch in enumerate(self.data_loader):
            # (input_ids, input_mask, segment_ids, ng_token_mask, target, deps, task)
            batch = {label: t.to(self.device, non_blocking=True) for label, t in batch.items()}
            current_step = (epoch - 1) * len(self.data_loader) + step

            loss, *_ = self.model(**batch, progress=current_step / self.total_step)

            if len(loss.size()) > 0:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss_value = loss.item()  # mean loss for this step
            total_loss += loss_value * next(iter(batch.values())).size(0)

            if step % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Time: {} Loss: {:.6f}'.format(
                    epoch,
                    step * self.data_loader.batch_size,
                    len(self.data_loader.dataset),
                    100.0 * step / len(self.data_loader),
                    datetime.datetime.now().strftime('%H:%M:%S'),
                    loss_value))

            gradient_accumulation_steps = self.gradient_accumulation_steps
            if step >= (len(self.data_loader) // self.gradient_accumulation_steps) * self.gradient_accumulation_steps:
                gradient_accumulation_steps = len(self.data_loader) % self.gradient_accumulation_steps  # fraction
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps  # scale loss
            loss.backward()
            accum_count += 1
            if accum_count == gradient_accumulation_steps:
                self.writer.set_step(self.writer.step + 1)
                self.writer.add_scalar('lr', self.lr_scheduler.get_last_lr()[0])
                self.writer.add_scalar('loss', loss_value)
                self.writer.add_scalar('progress', current_step / self.total_step)

                self.optimizer.step()
                self.optimizer.zero_grad()
                accum_count = 0
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

        log = {
            'loss': total_loss / len(self.data_loader.dataset),
        }
        corpus2result = {}
        for corpus, valid_data_loader in self.valid_data_loaders.items():
            val_log = self._valid_epoch(valid_data_loader, corpus)
            log[f'val_{corpus}_loss'] = val_log['loss']
            corpus2result[corpus] = val_log['result']
            if 'all' not in corpus2result:
                corpus2result['all'] = val_log['result']
            else:
                corpus2result['all'] += val_log['result']
        for corpus, result in corpus2result.items():
            val_metrics = self._eval_metrics(result.to_dict(), corpus)
            log.update({f'val_{corpus}_{met.__name__}': v for met, v in zip(self.metrics, val_metrics)})

        return log

    def _valid_epoch(self, data_loader, corpus):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_loss = 0
        arguments_set: List[List[List[int]]] = []
        contingency_set: List[int] = []
        with torch.no_grad():
            for step, batch in enumerate(data_loader):
                batch = {label: t.to(self.device, non_blocking=True) for label, t in batch.items()}

                loss, *output = self.model(**batch)

                if len(loss.size()) > 0:
                    loss = loss.mean()
                pas_scores = output[0]  # (b, seq, case, seq)

                if corpus != 'commonsense':
                    arguments_set += torch.argmax(pas_scores, dim=3).tolist()  # (b, seq, case)

                total_loss += loss.item() * pas_scores.size(0)

                if step % self.log_step == 0:
                    self.logger.info('Validation [{}/{} ({:.0f}%)] Time: {}'.format(
                        step * data_loader.batch_size,
                        len(data_loader.dataset),
                        100.0 * step / len(data_loader),
                        datetime.datetime.now().strftime('%H:%M:%S')))

        log = {'loss': total_loss / len(data_loader.dataset)}
        self.writer.add_scalar(f'loss/{corpus}', log['loss'])

        if corpus != 'commonsense':
            dataset = data_loader.dataset
            prediction_writer = PredictionKNPWriter(dataset, self.logger)
            documents_pred = prediction_writer.write(arguments_set, None, add_pas_tag=False)
            targets2label = {tuple(): '', ('pred',): 'pred', ('noun',): 'noun', ('pred', 'noun'): 'all'}

            scorer = Scorer(documents_pred, dataset.gold_documents,
                            target_cases=dataset.target_cases,
                            target_exophors=dataset.target_exophors,
                            coreference=dataset.coreference,
                            bridging=dataset.bridging,
                            pas_target=targets2label[tuple(dataset.pas_targets)])
            result = scorer.run()
            log['result'] = result
        else:
            log['f1'] = self._eval_commonsense(contingency_set)

        return log

    def _eval_commonsense(self, contingency_set: List[int]) -> float:
        valid_data_loader = self.valid_data_loaders['commonsense']
        gold = [f.label for f in valid_data_loader.dataset.features]
        f1 = f1_score(gold, contingency_set)
        self.writer.add_scalar(f'commonsense_f1', f1)
        return f1

    def _eval_metrics(self, result: dict, corpus: str):
        f1_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            f1_metrics[i] += metric(result)
            self.writer.add_scalar(f'{metric.__name__}/{corpus}', f1_metrics[i])
        return f1_metrics
