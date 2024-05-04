import logging
import warnings
from collections import defaultdict
from collections.abc import Mapping
from functools import reduce
from operator import add
from pathlib import Path

import hydra
import lightning.pytorch as pl
import torch
import transformers.utils.logging as hf_logging
from cohesion_tools.evaluators.cohesion import CohesionEvaluator, CohesionScore
from cohesion_tools.extractors import PasExtractor
from cohesion_tools.task import Task
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.utilities.warnings import PossibleUserWarning
from omegaconf import DictConfig, ListConfig, OmegaConf
from rhoknp import Document

from callbacks.prediction_writer import CohesionWriter
from datamodule.datamodule import CohesionDataModule
from datamodule.dataset.cohesion import CohesionDataset
from modules import CohesionModule
from utils.util import current_datetime_string

hf_logging.set_verbosity(hf_logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message=r"It is recommended to use .+ when logging on epoch level in distributed setting to accumulate the metric"
    r" across devices",
    category=PossibleUserWarning,
)
logging.getLogger("torch").setLevel(logging.WARNING)
OmegaConf.register_new_resolver("now", current_datetime_string, replace=True, use_cache=True)
OmegaConf.register_new_resolver("len", len, replace=True, use_cache=True)


@hydra.main(config_path="../configs", config_name="test", version_base=None)
def main(eval_cfg: DictConfig):
    if isinstance(eval_cfg.devices, str):
        eval_cfg.devices = (
            list(map(int, eval_cfg.devices.split(","))) if "," in eval_cfg.devices else int(eval_cfg.devices)
        )
    if isinstance(eval_cfg.max_batches_per_device, str):
        eval_cfg.max_batches_per_device = int(eval_cfg.max_batches_per_device)
    if isinstance(eval_cfg.num_workers, str):
        eval_cfg.num_workers = int(eval_cfg.num_workers)

    # Load saved model and config
    model = CohesionModule.load_from_checkpoint(checkpoint_path=hydra.utils.to_absolute_path(eval_cfg.checkpoint))
    if eval_cfg.compile is True:
        model = torch.compile(model)

    train_cfg: DictConfig = model.hparams
    OmegaConf.set_struct(train_cfg, False)  # enable to add new key-value pairs
    cfg = OmegaConf.merge(train_cfg, eval_cfg)
    assert isinstance(cfg, DictConfig)

    prediction_writer = CohesionWriter(analysis_target_threshold=cfg.analysis_target_threshold)

    num_devices: int = len(cfg.devices) if isinstance(cfg.devices, (list, ListConfig)) else cfg.devices
    cfg.effective_batch_size = cfg.max_batches_per_device * num_devices
    cfg.datamodule.batch_size = cfg.max_batches_per_device

    # Instantiate lightning trainer
    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=[prediction_writer],
        logger=False,
        devices=cfg.devices,
    )

    # Instantiate lightning datamodule
    datamodule = CohesionDataModule(cfg=cfg.datamodule)
    datamodule.setup(stage=TrainerFn.TESTING)

    datasets: dict[str, CohesionDataset]
    if cfg.eval_set == "valid":
        datasets = datamodule.valid_datasets
        dataloaders = datamodule.val_dataloader()
    elif cfg.eval_set == "test":
        datasets = datamodule.test_datasets
        dataloaders = datamodule.test_dataloader()
    else:
        raise ValueError(f"datasets for eval set {cfg.eval_set} not found")

    # Run evaluation
    raw_results: list[Mapping[str, float]] = trainer.test(model=model, dataloaders=dataloaders)
    save_results(raw_results, Path(cfg.eval_dir))

    # Run prediction
    pred_dir = Path(cfg.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    for corpus, dataloader in dataloaders.items():
        prediction_writer.knp_destination = pred_dir / f"knp_{corpus}"
        prediction_writer.json_destination = pred_dir / f"json_{corpus}"
        trainer.predict(model=model, dataloaders=dataloader)
    save_prediction(datasets, pred_dir)


@rank_zero_only
def save_results(results: list[Mapping[str, float]], save_dir: Path) -> None:
    test_results: dict[str, dict[str, float]] = defaultdict(dict)
    for k, v in [item for result in results for item in result.items()]:
        met, corpus = k.split("/")
        if met in test_results[corpus]:
            assert v == test_results[corpus][met]
        else:
            test_results[corpus][met] = v

    save_dir.mkdir(exist_ok=True, parents=True)
    for corpus, result in test_results.items():
        with save_dir.joinpath(f"{corpus}.csv").open(mode="wt") as f:
            f.write(",".join(result.keys()) + "\n")
            f.write(",".join(f"{v:.6}" for v in result.values()) + "\n")


@rank_zero_only
def save_prediction(datasets: dict[str, CohesionDataset], pred_dir: Path) -> None:
    all_results = []
    for corpus, dataset in datasets.items():
        predicted_documents: list[Document] = []
        for path in pred_dir.joinpath(f"knp_{corpus}").glob("*.knp"):
            predicted_documents.append(Document.from_knp(path.read_text()))
        pas_extractor = dataset.task_to_extractor[Task.PAS_ANALYSIS]
        assert isinstance(pas_extractor, PasExtractor)
        evaluator = CohesionEvaluator(
            tasks=dataset.tasks,
            exophora_referent_types=[e.type for e in dataset.exophora_referents],
            pas_cases=dataset.cases,
            bridging_rel_types=dataset.bar_rels,
        )
        score_result: CohesionScore = evaluator.run(
            gold_documents=dataset.orig_documents,
            predicted_documents=predicted_documents,
        )
        score_result.export_csv(pred_dir / f"{corpus}.csv")
        score_result.export_txt(pred_dir / f"{corpus}.txt")
        all_results.append(score_result)
    score_result = reduce(add, all_results)
    score_result.export_csv(pred_dir / "all.csv")
    score_result.export_txt(pred_dir / "all.txt")


if __name__ == "__main__":
    main()
