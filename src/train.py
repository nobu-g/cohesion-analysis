import logging
import math
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Union

import hydra
import lightning.pytorch as pl
import torch
import transformers.utils.logging as hf_logging
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities.warnings import PossibleUserWarning
from omegaconf import DictConfig, ListConfig, OmegaConf

from datamodule.datamodule import CohesionDataModule
from modules import CohesionModule
from test import save_results
from utils.util import current_datetime_string

hf_logging.set_verbosity(hf_logging.ERROR)
logging.getLogger("rhoknp").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    r"It is recommended to use .+ when logging on epoch level in distributed setting to accumulate the metric across "
    r"devices\.",
    category=PossibleUserWarning,
)
warnings.filterwarnings(
    "ignore",
    r"The dataloader, .+, does not have many workers which may be a bottleneck\. Consider increasing the value of the "
    r"`num_workers` argument` \(try .+ which is the number of cpus on this machine\) in the `DataLoader` init to "
    r"improve performance\.",
    category=PossibleUserWarning,
)
OmegaConf.register_new_resolver("now", current_datetime_string, replace=True, use_cache=True)
OmegaConf.register_new_resolver("len", len, replace=True, use_cache=True)


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    if isinstance(cfg.devices, str):
        cfg.devices = list(map(int, cfg.devices.split(","))) if "," in cfg.devices else int(cfg.devices)
    if isinstance(cfg.max_batches_per_device, str):
        cfg.max_batches_per_device = int(cfg.max_batches_per_device)
    if isinstance(cfg.num_workers, str):
        cfg.num_workers = int(cfg.num_workers)
    cfg.seed = pl.seed_everything(seed=cfg.seed, workers=True)

    # Instantiate lightning module
    model = CohesionModule(hparams=cfg)
    if cfg.checkpoint:
        model.load_from_checkpoint(checkpoint_path=hydra.utils.to_absolute_path(cfg.checkpoint))
    if cfg.compile is True:
        model = torch.compile(model)  # type: ignore[assignment]

    # Instantiate lightning loggers
    logger: Union[Logger, bool] = cfg.get("logger", False) and hydra.utils.instantiate(cfg.get("logger"))
    # Instantiate lightning callbacks
    callbacks: list[Callback] = list(map(hydra.utils.instantiate, cfg.get("callbacks", {}).values()))

    # Calculate gradient_accumulation_steps assuming DDP
    num_devices: int = 1
    if isinstance(cfg.devices, (list, ListConfig)):
        num_devices = len(cfg.devices)
    elif isinstance(cfg.devices, int):
        num_devices = cfg.devices
    cfg.trainer.accumulate_grad_batches = math.ceil(
        cfg.effective_batch_size / (cfg.max_batches_per_device * num_devices),
    )
    batches_per_device = cfg.effective_batch_size // (num_devices * cfg.trainer.accumulate_grad_batches)
    # if effective_batch_size % (accumulate_grad_batches * num_devices) != 0, then
    # cause an error of at most accumulate_grad_batches * num_devices compared in effective_batch_size
    # otherwise, no error
    cfg.effective_batch_size = batches_per_device * num_devices * cfg.trainer.accumulate_grad_batches
    cfg.datamodule.batch_size = batches_per_device

    # Instantiate lightning trainer
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks, devices=cfg.devices)

    # Instantiate lightning datamodule
    datamodule = CohesionDataModule(cfg=cfg.datamodule)

    # Run training
    trainer.fit(model=model, datamodule=datamodule)

    # Run test
    raw_results: list[Mapping[str, float]] = trainer.test(model=model, datamodule=datamodule)
    save_results(raw_results, Path(cfg.run_dir) / "eval_test")

    wandb.finish()


if __name__ == "__main__":
    main()
