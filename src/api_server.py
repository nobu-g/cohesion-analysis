import copy
import logging
import tempfile
from pathlib import Path

import hydra
import uvicorn
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import DictConfig
from rhoknp import Document

from datamodule.dataset import CohesionDataset
from datamodule.example.kyoto import Task
from predict import Analyzer
from server.response import gen_response
from utils.util import current_datetime_string
from writer.json import DocumentProb, Phrase

logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analysis")
async def analysis(request: Request):
    body_json = await request.json()
    log_dir = Path("log") / current_datetime_string(r"%Y%m%d_%H%M%S")

    analyzer: Analyzer = app.state.analyzer
    document = analyzer.gen_document_from_raw_text(body_json["value"])
    with tempfile.TemporaryDirectory() as input_dir:
        Path(input_dir).joinpath(f"{document.doc_id}.knp").write_text(document.to_knp())
        dataloader = analyzer.gen_dataloader(Path(input_dir))
    analyzer.analyze(dataloader, knp_destination=log_dir, json_destination=log_dir)

    dataset = dataloader.dataset
    assert isinstance(dataset, CohesionDataset)
    analyzed_document = Document.from_knp(log_dir.joinpath(f"{document.doc_id}.knp").read_text())
    document_prob = DocumentProb.from_json(log_dir.joinpath(f"{document.doc_id}.json").read_text())
    response = gen_response(analyzed_document, dataset.task_to_rels, document_prob)
    return jsonable_encoder(response)


@app.post("/system")
async def system(request: Request):
    body_json = await request.json()
    doc_id, corpus, split = body_json["doc_id"], body_json["corpus"], body_json["split"]
    cfg: DictConfig = app.state.cfg
    dataset_cfg = cfg.datamodule.predict
    pred_dir = Path(cfg.run_dir) / f"pred_{split}"
    document = Document.from_knp(pred_dir.joinpath(f"knp_{corpus}/{doc_id}.knp").read_text())
    task_to_rels: dict[Task, list[str]] = {
        Task.PAS_ANALYSIS: list(dataset_cfg.cases),
        Task.BRIDGING_REFERENCE_RESOLUTION: list(dataset_cfg.bar_rels),
        Task.COREFERENCE_RESOLUTION: ["="],
    }
    prediction = DocumentProb.from_json(pred_dir.joinpath(f"json_{corpus}/{doc_id}.json").read_text())
    response = gen_response(document, task_to_rels, prediction)
    return jsonable_encoder(response)


@app.post("/gold")
async def gold(request: Request):
    body_json = await request.json()
    doc_id, corpus, split = body_json["doc_id"], body_json["corpus"], body_json["split"]
    cfg: DictConfig = app.state.cfg
    dataset_cfg = copy.deepcopy(getattr(getattr(cfg.datamodule, split), corpus))
    dataset_cfg.knp_path = dataset_cfg.knp_path + f"/{doc_id}.knp"
    dataset_cfg.max_seq_length = 2048
    dataset = hydra.utils.instantiate(dataset_cfg)
    assert len(dataset.examples) == len(dataset.documents) == 1
    pred_dir = Path(cfg.run_dir) / f"pred_{split}"
    prediction = DocumentProb.from_json(pred_dir.joinpath(f"json_{corpus}/{doc_id}.json").read_text())
    response = gen_response(dataset.documents[0], dataset.task_to_rels, prediction)
    return jsonable_encoder(response)


@app.post("/knp")
async def knp(request: Request):
    body_json = await request.json()
    path = body_json["path"]
    doc_id = body_json["doc_id"]
    cfg: DictConfig = app.state.cfg
    dataset_cfg = cfg.datamodule.predict
    document = Document.from_knp(Path(path).joinpath(f"{doc_id}.knp").read_text())
    task_to_rel_types = {
        Task.PAS_ANALYSIS: list(dataset_cfg.cases),
        Task.BRIDGING_REFERENCE_RESOLUTION: list(dataset_cfg.bar_rels),
        Task.COREFERENCE_RESOLUTION: ["="],
    }
    response = gen_response(
        document,
        task_to_rel_types,
        DocumentProb(
            document.doc_id,
            list(cfg.special_tokens),
            [Phrase([]) for _ in document.base_phrases],
        ),
    )
    return jsonable_encoder(response)


@app.post("/dids")
async def doc_ids(request: Request) -> list[str]:
    body_json = await request.json()
    corpus, split = body_json["corpus"], body_json["split"]
    cfg: DictConfig = request.app.state.cfg
    paths = sorted(Path(cfg.run_dir).joinpath(f"pred_{split}", f"knp_{corpus}").glob("*.knp"))
    return [path.stem for path in paths]


@app.post("/corpora")
async def corpora(request: Request) -> list[str]:
    body_json = await request.json()
    split = body_json["split"]
    cfg: DictConfig = request.app.state.cfg
    corpora_paths = Path(cfg.run_dir).joinpath(f"pred_{split}").glob("knp_*")
    return [p.name[4:] for p in corpora_paths if p.is_dir()]


@app.get("/reltypes")
async def reltypes() -> list[str]:
    cfg: DictConfig = app.state.cfg
    dataset_cfg = cfg.datamodule.predict
    task_to_rel_types = {
        Task.PAS_ANALYSIS: list(dataset_cfg.cases),
        Task.BRIDGING_REFERENCE_RESOLUTION: list(dataset_cfg.bar_rels),
        Task.COREFERENCE_RESOLUTION: ["="],
    }
    return sum(task_to_rel_types.values(), [])


@app.get("/models")
async def models() -> list[str]:
    cfg: DictConfig = app.state.cfg
    return [m.name for m in cfg.models]


@hydra.main(config_path="../configs", config_name="server", version_base=None)
def main(cfg: DictConfig):
    if cfg.devices:
        if isinstance(cfg.devices, str):
            if "," in cfg.devices:
                cfg.devices = [int(x) for x in cfg.devices.split(",")]
            else:
                cfg.devices = int(cfg.devices)
    else:
        cfg.devices = 1

    analyzer = Analyzer(cfg)
    app.state.analyzer = analyzer
    app.state.cfg = analyzer.cfg
    uvicorn.run(app, host=cfg.host, port=cfg.port, workers=0, log_level="info")


if __name__ == "__main__":
    main()
