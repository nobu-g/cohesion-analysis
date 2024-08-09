from functools import reduce
from pathlib import Path
from statistics import mean
from typing import Any, Optional

import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig
from torch import nn
from typing_extensions import override

from metrics import CohesionMetric
from modules.model.loss import binary_cross_entropy_with_logits, cross_entropy_loss
from utils.util import IGNORE_INDEX, oc_resolve


class CohesionModule(pl.LightningModule):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__()
        oc_resolve(hparams, keys=hparams.keys_to_resolve)
        # this line allows accessing init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(hparams)
        self.valid_corpora = list(hparams.datamodule.valid.keys())
        self.test_corpora = list(hparams.datamodule.test.keys())
        self.val_metrics: dict[str, CohesionMetric] = {
            corpus: CohesionMetric(hparams.analysis_target_threshold) for corpus in self.valid_corpora
        }
        self.test_metrics: dict[str, CohesionMetric] = {
            corpus: CohesionMetric(hparams.analysis_target_threshold) for corpus in self.test_corpora
        }

        self.model: nn.Module = hydra.utils.instantiate(
            hparams.model,
            num_relations=int("pas" in hparams.tasks) * len(hparams.cases)
            + int("coreference" in hparams.tasks)
            + int("bridging" in hparams.tasks),
        )

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def configure_optimizers(self):
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ("bias", "LayerNorm.weight")
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.hparams.optimizer.weight_decay,
                "name": "decay",
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
                "name": "no_decay",
            },
        ]
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer,
            params=optimizer_grouped_parameters,
            _convert_="partial",
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.hparams.warmup_steps or total_steps * self.hparams.warmup_ratio
        if hasattr(self.hparams.scheduler, "num_warmup_steps"):
            self.hparams.scheduler.num_warmup_steps = warmup_steps
        if hasattr(self.hparams.scheduler, "num_training_steps"):
            self.hparams.scheduler.num_training_steps = total_steps
        lr_scheduler = hydra.utils.instantiate(self.hparams.scheduler, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        relation_logits, source_mask_logits = self.model(**batch)
        return {
            "relation_logits": relation_logits.masked_fill(~batch["target_mask"], -1024.0),
            "source_mask_logits": source_mask_logits,
        }

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ret: dict[str, torch.Tensor] = self(batch)
        losses: dict[str, torch.Tensor] = {}

        source_mask: torch.Tensor = batch["source_mask"]  # (b, seq)
        target_mask: torch.Tensor = batch["target_mask"]  # (b, rel, seq, seq)

        relation_mask = source_mask.unsqueeze(1).unsqueeze(3) & target_mask  # (b, rel, seq, seq)
        losses["relation_loss"] = cross_entropy_loss(ret["relation_logits"], batch["target_label"], relation_mask)

        source_label: torch.Tensor = batch["source_label"]  # (b, task, seq)
        analysis_target_mask = source_label.ne(IGNORE_INDEX) & source_mask.unsqueeze(1)  # (b, task, seq)
        source_label = torch.where(analysis_target_mask, source_label, torch.zeros_like(source_label))
        losses["source_mask_loss"] = binary_cross_entropy_with_logits(
            ret["source_mask_logits"], source_label, analysis_target_mask
        )
        # weighted sum
        losses["loss"] = losses["relation_loss"] + losses["source_mask_loss"] * 0.5
        self.log_dict({f"train/{key}": value for key, value in losses.items()})
        return losses["loss"]

    @override
    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        prediction = self.predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        metric = self.val_metrics[self.valid_corpora[dataloader_idx]]
        metric.update(prediction)

    @override
    def on_validation_epoch_end(self) -> None:
        metrics_log: dict[str, dict[str, float]] = {}
        for corpus in self.valid_corpora:
            metric = self.val_metrics[corpus]
            assert isinstance(self.trainer.val_dataloaders, dict)
            metric.set_attributes(dataset=self.trainer.val_dataloaders[corpus].dataset)
            metrics = metric.compute()
            metrics_log[corpus] = metrics
            metric.reset()

        for corpus, metrics in metrics_log.items():
            self.log_dict({f"valid_{corpus}/{key}": value for key, value in metrics.items()})
        for key in reduce(set.union, [set(metrics.keys()) for metrics in metrics_log.values()]):
            mean_score = mean(metrics_log[corpus][key] for corpus in self.valid_corpora if key in metrics_log[corpus])
            self.log(f"valid/{key}", mean_score)
        for key in reduce(set.union, [set(metrics.keys()) for metrics in metrics_log.values()]):
            mean_score = mean(
                metrics_log[corpus][key]
                for corpus in self.valid_corpora
                if key in metrics_log[corpus] and corpus != "fuman"
            )
            self.log(f"valid_wo_fuman/{key}", mean_score)

    @rank_zero_only
    def on_train_end(self) -> None:
        best_model_path: str = self.trainer.checkpoint_callback.best_model_path  # type: ignore[union-attr]
        if not best_model_path:
            return
        save_dir = Path(self.hparams.exp_dir) / self.hparams.run_id  # type: ignore[attr-defined]
        best_path = save_dir / "best.ckpt"
        if best_path.exists():
            best_path.unlink()
        actual_best_path = Path(best_model_path)
        assert actual_best_path.parent.resolve() == best_path.parent.resolve()
        best_path.resolve().symlink_to(actual_best_path.name)

    @override
    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        prediction = self.predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        metric = self.test_metrics[self.test_corpora[dataloader_idx]]
        metric.update(prediction)

    @override
    def on_test_epoch_end(self) -> None:
        metrics_log = {}
        for corpus in self.test_corpora:
            metric = self.test_metrics[corpus]
            assert isinstance(self.trainer.test_dataloaders, dict)
            metric.set_attributes(dataset=self.trainer.test_dataloaders[corpus].dataset)
            metrics = metric.compute()
            metrics_log[corpus] = metrics
            metric.reset()

        for corpus, metrics in metrics_log.items():
            self.log_dict({f"test_{corpus}/{key}": value for key, value in metrics.items()})
        for key in reduce(set.union, [set(metrics.keys()) for metrics in metrics_log.values()]):
            mean_score = mean(metrics_log[corpus][key] for corpus in self.test_corpora if key in metrics_log[corpus])
            self.log(f"test/{key}", mean_score)
        for key in reduce(set.union, [set(metrics.keys()) for metrics in metrics_log.values()]):
            mean_score = mean(
                metrics_log[corpus][key]
                for corpus in self.test_corpora
                if key in metrics_log[corpus] and corpus != "fuman"
            )
            self.log(f"test_wo_fuman/{key}", mean_score)

    @override
    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> dict[str, Any]:
        output: dict[str, torch.Tensor] = self(batch)
        return {"example_ids": batch["example_id"], "dataloader_idx": dataloader_idx, **output}
