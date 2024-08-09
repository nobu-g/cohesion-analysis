import logging
import sys
import tempfile
from pathlib import Path
from typing import TextIO, Union

import hydra
import jaconv
import lightning.pytorch as pl
import torch
import transformers.utils.logging as hf_logging
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer.states import TrainerFn
from omegaconf import DictConfig, OmegaConf
from rhoknp import KNP, KWJA, Document
from torch.utils.data import DataLoader

from callbacks.prediction_writer import CohesionWriter
from datamodule.datamodule import CohesionDataModule
from modules import CohesionModule
from utils.util import current_datetime_string

hf_logging.set_verbosity(hf_logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

OmegaConf.register_new_resolver("now", current_datetime_string, replace=True, use_cache=True)
OmegaConf.register_new_resolver("len", len, replace=True, use_cache=True)


class Analyzer:
    def __init__(self, cfg: DictConfig) -> None:
        # Load saved model and config
        self.device = self._prepare_device(device_name="auto")
        self.model = CohesionModule.load_from_checkpoint(
            checkpoint_path=hydra.utils.to_absolute_path(cfg.checkpoint),
            map_location=self.device,
        )
        cfg_train: DictConfig = self.model.hparams  # type: ignore[assignment]
        OmegaConf.set_struct(cfg_train, False)  # enable to add new key-value pairs
        self.cfg = OmegaConf.merge(cfg_train, cfg)
        assert isinstance(self.cfg, DictConfig)

        callbacks: list[Callback] = list(map(hydra.utils.instantiate, self.cfg.get("callbacks", {}).values()))
        self.prediction_writer = CohesionWriter(analysis_target_threshold=self.cfg.analysis_target_threshold)

        # Instantiate lightning trainer
        self.trainer: pl.Trainer = hydra.utils.instantiate(
            self.cfg.trainer,
            callbacks=[*callbacks, self.prediction_writer],
            logger=False,
            devices=self.cfg.devices,
        )

    @staticmethod
    def _prepare_device(device_name: str) -> torch.device:
        n_gpu = torch.cuda.device_count()
        if device_name == "auto":
            if n_gpu > 0:
                device_name = "gpu"
            else:
                device_name = "cpu"
        if device_name == "gpu" and n_gpu == 0:
            logger.warning("There's no GPU available on this machine. Using CPU instead.")
            return torch.device("cpu")
        else:
            return torch.device("cuda:0" if device_name == "gpu" else "cpu")

    def gen_document_from_raw_text(self, raw_text: str) -> Document:
        raw_document = Document.from_raw_text(self.sanitize_string(raw_text))
        logger.info("input text: " + raw_document.text)
        knp = KNP(options=["-tab", "-disable-segmentation-modification", "-dpnd-fast"])
        if knp.is_available():
            document = knp.apply_to_document(raw_document)
        else:
            kwja = KWJA(options=["--tasks", "char,word"])
            document = kwja.apply_to_document(raw_document)
        document.doc_id = "0"
        for idx, sentence in enumerate(document.sentences):
            sentence.sid = f"0-{idx + 1}"
        return document

    def gen_dataloader(self, input_knp_path: Path) -> DataLoader:
        # Instantiate lightning datamodule
        datamodule_cfg = self.cfg.datamodule
        datamodule_cfg.predict.knp_path = str(input_knp_path)
        datamodule_cfg.num_workers = int(self.cfg.num_workers)
        datamodule_cfg.batch_size = int(self.cfg.max_batches_per_device)
        datamodule = CohesionDataModule(cfg=datamodule_cfg)
        datamodule.setup(stage=TrainerFn.PREDICTING)
        return datamodule.predict_dataloader()

    def analyze(
        self,
        dataloader: DataLoader,
        knp_destination: Union[Path, TextIO, None] = None,
        json_destination: Union[Path, TextIO, None] = None,
    ) -> None:
        self.prediction_writer.knp_destination = knp_destination
        self.prediction_writer.json_destination = json_destination
        self.trainer.predict(model=self.model, dataloaders=dataloader)

    @staticmethod
    def sanitize_string(string: str) -> str:
        string = "".join(string.split())  # remove space character
        # Although Juman++/KNP support hankaku, the training datasets are normalized to zenkaku.
        string = jaconv.h2z(string, digit=True, ascii=True)
        return string


@hydra.main(config_path="../configs", config_name="predict", version_base=None)
def main(cfg: DictConfig):
    if isinstance(cfg.devices, str):
        cfg.devices = list(map(int, cfg.devices.split(","))) if "," in cfg.devices else int(cfg.devices)
    if isinstance(cfg.max_batches_per_device, str):
        cfg.max_batches_per_device = int(cfg.max_batches_per_device)
    if isinstance(cfg.num_workers, str):
        cfg.num_workers = int(cfg.num_workers)

    analyzer = Analyzer(cfg)

    source: Union[Path, str]
    if cfg.input_file is not None:
        source = Path(cfg.input_file).read_text()
    elif cfg.input_knp is not None:
        source = Path(cfg.input_knp)
    else:
        source = "".join(sys.stdin.readlines())

    destination: Union[Path, TextIO, None]
    if cfg.export_dir is not None:
        destination = Path(cfg.export_dir)
    else:
        destination = sys.stdout

    if isinstance(source, str):
        document = analyzer.gen_document_from_raw_text(source)
        with tempfile.TemporaryDirectory() as input_dir:
            Path(input_dir).joinpath(f"{document.doc_id}.knp").write_text(document.to_knp())
            dataloader = analyzer.gen_dataloader(Path(input_dir))
    else:
        dataloader = analyzer.gen_dataloader(source)
    analyzer.analyze(dataloader, knp_destination=destination)


if __name__ == "__main__":
    main()
