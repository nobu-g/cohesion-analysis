import re
import argparse
from pathlib import Path
from typing import List, Callable, Set, Dict

import numpy as np
import torch.nn as nn
from sklearn.metrics import f1_score

import data_loader.data_loaders as module_loader
import data_loader.dataset as module_dataset
import model.metric as module_metric
import model.model as module_arch
from data_loader.dataset import PASDataset
from utils.parse_config import ConfigParser
from prediction.prediction_writer import PredictionKNPWriter
from prediction.inference import Inference
from scorer import Scorer, ScoreResult


class Tester:
    def __init__(self, model, metrics, config, data_loaders, eval_set, logger, predict_overt, precision_threshold,
                 recall_threshold, result_suffix):
        self.model: nn.Module = model
        self.metrics: List[Callable] = metrics
        self.config = config
        self.data_loaders = data_loaders
        self.eval_set: str = eval_set
        self.logger = logger
        self.predict_overt: bool = predict_overt

        self.inference = Inference(config, model,
                                   precision_threshold=precision_threshold,
                                   recall_threshold=recall_threshold,
                                   logger=logger)

        self.save_dir: Path = config.save_dir / f'eval_{eval_set}{result_suffix}'
        self.save_dir.mkdir(exist_ok=True)
        pas_targets: Set[str] = set()
        for corpus, data_loader in self.data_loaders.items():
            if corpus != 'commonsense':
                pas_targets |= set(data_loader.dataset.pas_targets)
        self.pas_targets: List[str] = []
        if 'pred' in pas_targets:
            self.pas_targets.append('pred')
        if 'noun' in pas_targets:
            self.pas_targets.append('noun')
        if 'noun' in pas_targets and 'pred' in pas_targets:
            self.pas_targets.append('all')
        if not self.pas_targets:
            self.pas_targets.append('')

    def test(self):
        log = {}
        all_result: Dict[str, ScoreResult] = {}
        for corpus, data_loader in self.data_loaders.items():
            corpus_result = self._test(data_loader, corpus)
            log[f'{self.eval_set}_{corpus}_loss'] = corpus_result.pop('loss')
            for pas_target, result in corpus_result.items():
                if pas_target not in all_result:
                    all_result[pas_target] = result
                else:
                    all_result[pas_target] += result
                for met, value in zip(self.metrics, self._eval_metrics(result.to_dict())):
                    met_name = met.__name__
                    if 'case_analysis' in met_name or 'zero_anaphora' in met_name:
                        met_name = f'{pas_target}_{met_name}'
                    log[f'{self.eval_set}_{corpus}_{met_name}'] = value

        for pas_target, result in all_result.items():
            for met, value in zip(self.metrics, self._eval_metrics(result.to_dict())):
                met_name = met.__name__
                if 'case_analysis' in met_name or 'zero_anaphora' in met_name:
                    met_name = f'{pas_target}_{met_name}'
                log[f'{self.eval_set}_all_{met_name}'] = value
            target = 'all' + (f'_{pas_target}' if pas_target else '')
            result.export_txt(self.save_dir / f'{target}.txt')
            result.export_csv(self.save_dir / f'{target}.csv')

        return log

    def _test(self, data_loader, corpus: str):

        loss, *predictions = self.inference(data_loader)

        log = {}
        if corpus != 'commonsense':
            prediction = predictions[0]  # (N, seq, case, seq)
            result = self._eval_pas(prediction.tolist(), data_loader.dataset, corpus=corpus)
        else:
            assert self.config['arch']['type'] == 'CommonsenseModel'
            result = self._eval_commonsense(predictions[1].tolist(), data_loader)
        log.update(result)
        log['loss'] = loss
        return log

    def _eval_pas(self, arguments_set, dataset: PASDataset, corpus: str, suffix: str = '') -> Dict[str, ScoreResult]:
        prediction_output_dir = self.save_dir / f'{corpus}_out{suffix}'
        prediction_writer = PredictionKNPWriter(dataset,
                                                self.logger,
                                                use_knp_overt=(not self.predict_overt))
        documents_pred = prediction_writer.write(arguments_set, prediction_output_dir, add_pas_tag=False)

        log = {}
        for pas_target in self.pas_targets:
            scorer = Scorer(documents_pred, dataset.gold_documents,
                            target_cases=dataset.target_cases,
                            target_exophors=dataset.target_exophors,
                            coreference=dataset.coreference,
                            bridging=dataset.bridging,
                            pas_target=pas_target)
            result = scorer.run()
            target = corpus + (f'_{pas_target}' if pas_target else '') + suffix

            scorer.write_html(self.save_dir / f'{target}.html')
            result.export_txt(self.save_dir / f'{target}.txt')
            result.export_csv(self.save_dir / f'{target}.csv')

            log[pas_target] = result

        return log

    def _eval_metrics(self, result: dict):
        f1_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            f1_metrics[i] += metric(result)
        return f1_metrics

    @staticmethod
    def _eval_commonsense(contingency_set: np.ndarray, data_loader) -> dict:
        assert data_loader.dataset.__class__.__name__ == 'CommonsenseDataset'
        gold = np.array([f.label for f in data_loader.dataset.features])
        return {'f1': f1_score(gold, contingency_set)}


def main(config, args):
    logger = config.get_logger(args.target)

    # setup data_loader instances
    data_loaders = {}
    for corpus in config[f'{args.target}_datasets'].keys():
        if args.oracle is True:
            assert config[f'{args.target}_datasets'][corpus]['args']['path'].endswith(args.target)
            config[f'{args.target}_datasets'][corpus]['args']['path'] += '_oracle'
        dataset = config.init_obj(f'{args.target}_datasets.{corpus}', module_dataset, logger=logger)
        data_loaders[corpus] = config.init_obj(f'data_loaders.{args.target}', module_loader, dataset)

    # build model architecture
    model: nn.Module = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of metrics
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    tester = Tester(model, metric_fns, config, data_loaders, args.target, logger, args.predict_overt,
                    args.precision_threshold, args.recall_threshold, args.result_suffix)

    log = tester.test()

    # print logged information to the screen
    key_max = max(len(s) for s in log.keys())
    for key, value in log.items():
        logger.info(f'{key:{key_max}s}: {value:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to checkpoint to test')
    parser.add_argument('--ens', '--ensemble', default=None, type=str,
                        help='path to directory where checkpoints to ensemble exist')
    parser.add_argument('-d', '--device', default='', type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-t', '--target', default='test', type=str, choices=['valid', 'test'],
                        help='evaluation target')
    parser.add_argument('--predict-overt', action='store_true', default=False,
                        help='calculate scores for overt arguments instead of using gold')
    parser.add_argument('--precision-threshold', default=0.0, type=float,
                        help='threshold for argument existence. The higher you set, the higher precision gets. [0, 1]')
    parser.add_argument('--recall-threshold', default=0.0, type=float,
                        help='threshold for argument non-existence. The higher you set, the higher recall gets [0, 1]')
    parser.add_argument('--result-suffix', default='', type=str,
                        help='custom evaluation result directory name')
    parser.add_argument('--run-id', default=None, type=str,
                        help='custom experiment directory name')
    parser.add_argument('--oracle', action='store_true', default=False,
                        help='use oracle dependency labels')
    parsed_args = parser.parse_args()
    inherit_save_dir = (parsed_args.resume is not None and parsed_args.run_id is None)
    main(ConfigParser.from_args(parsed_args, run_id=parsed_args.run_id, inherit_save_dir=inherit_save_dir), parsed_args)
