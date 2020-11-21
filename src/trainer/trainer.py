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


class Trainer(BaseTrainer):
    """
    Trainer class
    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, metrics, optimizer, config, train_dataset,
                 valid_kwdlc_dataset, valid_kc_dataset, valid_commonsense_dataset,
                 lr_scheduler=None):
        super().__init__(model, metrics, optimizer, config, train_dataset)
        self.config = config
        self.config['valid_data_loader']['args']['batch_size'] = self.data_loader.batch_size
        self.valid_kwdlc_data_loader = None
        self.valid_kc_data_loader = None
        self.valid_commonsense_data_loader = None
        if valid_kwdlc_dataset is not None:
            self.config['valid_data_loader']['args']['batch_size'] = self.data_loader.batch_size
            self.valid_kwdlc_data_loader = config.init_obj('valid_data_loader', module_loader, valid_kwdlc_dataset)
        if valid_kc_dataset is not None:
            self.valid_kc_data_loader = config.init_obj('valid_data_loader', module_loader, valid_kc_dataset)
        if valid_commonsense_dataset is not None:
            self.valid_commonsense_data_loader = config.init_obj('valid_data_loader', module_loader,
                                                                 valid_commonsense_dataset)

        self.lr_scheduler = lr_scheduler
        self.log_step = math.ceil(len(self.data_loader.dataset) / np.sqrt(self.data_loader.batch_size) / 200)

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
        for step, batch in enumerate(self.data_loader):
            # (input_ids, input_mask, segment_ids, ng_token_mask, target, deps, task)
            batch = {label: t.to(self.device, non_blocking=True) for label, t in batch.items()}
            current_step = (epoch - 1) * len(self.data_loader) + step

            loss, *_ = self.model(**batch, progress=current_step / self.total_step)

            if len(loss.size()) > 0:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss_value = loss.item()
            total_loss += loss_value * next(iter(batch.values())).size(0)

            if step % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Time: {} Loss: {:.6f}'.format(
                    epoch,
                    step * self.data_loader.batch_size,
                    len(self.data_loader.dataset),
                    100.0 * step / len(self.data_loader),
                    datetime.datetime.now().strftime('%H:%M:%S'),
                    loss_value))

            if step < (len(self.data_loader) // self.gradient_accumulation_steps) * self.gradient_accumulation_steps:
                gradient_accumulation_steps = self.gradient_accumulation_steps
            else:
                gradient_accumulation_steps = len(self.data_loader) % self.gradient_accumulation_steps
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0:
                self.writer.set_step(
                    (epoch - 1) * self.optimization_step_per_epoch + (step + 1) // gradient_accumulation_steps - 1)
                self.writer.add_scalar('lr', self.lr_scheduler.get_last_lr()[0])
                self.writer.add_scalar('loss', loss_value)
                self.writer.add_scalar('progress', current_step / self.total_step)

                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

        log = {
            'loss': total_loss / len(self.data_loader.dataset),
        }

        if self.valid_kwdlc_data_loader is not None:
            val_log = self._valid_epoch(epoch, self.valid_kwdlc_data_loader, 'kwdlc')
            log.update(**{'val_kwdlc_'+k: v for k, v in val_log.items()})

        if self.valid_kc_data_loader is not None:
            val_log = self._valid_epoch(epoch, self.valid_kc_data_loader, 'kc')
            log.update(**{'val_kc_'+k: v for k, v in val_log.items()})

        if self.valid_commonsense_data_loader is not None:
            val_log = self._valid_epoch(epoch, self.valid_commonsense_data_loader, 'commonsense')
            log.update(**{'val_commonsense_'+k: v for k, v in val_log.items()})

        return log

    def _valid_epoch(self, epoch, data_loader, label):
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

                if label in ('kwdlc', 'kc'):
                    arguments_set += torch.argmax(pas_scores, dim=3).tolist()  # (b, seq, case)

                total_loss += loss.item() * pas_scores.size(0)

                self.writer.set_step((epoch - 1) * len(data_loader) + step, 'valid')
                self.writer.add_scalar(f'loss_{label}', loss.item())

                if step % self.log_step == 0:
                    self.logger.info('Validation [{}/{} ({:.0f}%)] Time: {}'.format(
                        step * data_loader.batch_size,
                        len(self.data_loader.dataset),
                        100.0 * step / len(data_loader),
                        datetime.datetime.now().strftime('%H:%M:%S')))

        log = {'loss': total_loss / len(self.data_loader.dataset)}

        if label in ('kwdlc', 'kc'):
            prediction_writer = PredictionKNPWriter(data_loader.dataset, self.logger)
            documents_pred = prediction_writer.write(arguments_set, None)
            if label == 'kc':
                documents_gold = data_loader.dataset.joined_documents
            else:
                documents_gold = data_loader.dataset.documents
            targets2label = {tuple(): '', ('pred',): 'pred', ('noun',): 'noun', ('pred', 'noun'): 'all'}

            scorer = Scorer(documents_pred, documents_gold,
                            target_cases=data_loader.dataset.target_cases,
                            target_exophors=data_loader.dataset.target_exophors,
                            coreference=data_loader.dataset.coreference,
                            bridging=data_loader.dataset.bridging,
                            pas_target=targets2label[tuple(data_loader.dataset.pas_targets)])

            val_metrics = self._eval_metrics(scorer.result_dict(), label)

            log.update(dict(zip([met.__name__ for met in self.metrics], val_metrics)))
        elif label == 'commonsense':
            log['f1'] = self._eval_commonsense(contingency_set)

        return log

    def _eval_commonsense(self, contingency_set: List[int]) -> float:
        gold = [f.label for f in self.valid_commonsense_data_loader.dataset.features]
        f1 = f1_score(gold, contingency_set)
        self.writer.add_scalar(f'commonsense_f1', f1)
        return f1

    def _eval_metrics(self, result: dict, label: str):
        f1_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            f1_metrics[i] += metric(result)
            self.writer.add_scalar(f'{label}_{metric.__name__}', f1_metrics[i])
        return f1_metrics
