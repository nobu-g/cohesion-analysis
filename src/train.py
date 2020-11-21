import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
import transformers.optimization as module_optim

import data_loader.dataset as module_dataset
import model.metric as module_metric
import model.model as module_arch
from utils.parse_config import ConfigParser
from trainer import Trainer


def main(config: ConfigParser, args: argparse.Namespace):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    # torch.autograd.set_detect_anomaly(True)

    logger = config.get_logger('train')

    # setup data_loader instances
    train_datasets = []
    if config['train_kwdlc_dataset'] is not None:
        train_datasets.append(config.init_obj('train_kwdlc_dataset', module_dataset, logger=logger))
    if config['train_kc_dataset'] is not None:
        train_datasets.append(config.init_obj('train_kc_dataset', module_dataset, logger=logger))
    if config['train_commonsense_dataset'] is not None:
        train_datasets.append(config.init_obj('train_commonsense_dataset', module_dataset, logger=logger))
    train_dataset = ConcatDataset(train_datasets)
    valid_kwdlc_dataset = None
    if config['valid_kwdlc_dataset'] is not None:
        valid_kwdlc_dataset = config.init_obj('valid_kwdlc_dataset', module_dataset, logger=logger)
    valid_kc_dataset = None
    if config['valid_kc_dataset'] is not None:
        valid_kc_dataset = config.init_obj('valid_kc_dataset', module_dataset, logger=logger)
    valid_commonsense_dataset = None
    if config['valid_commonsense_dataset'] is not None:
        valid_commonsense_dataset = config.init_obj('valid_commonsense_dataset', module_dataset, logger=logger)

    # build model architecture, then print to console
    model: nn.Module = config.init_obj('arch', module_arch, vocab_size=train_datasets[0].expanded_vocab_size)
    logger.info(model)

    # get function handles of metrics
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler
    trainable_named_params = filter(lambda x: x[1].requires_grad, model.named_parameters())
    no_decay = ('bias', 'LayerNorm.weight')
    optimizer_grouped_parameters = [
        {'params': [p for n, p in trainable_named_params if not any(nd in n for nd in no_decay)],
         'weight_decay': config['optimizer']['args']['weight_decay']},
        {'params': [p for n, p in trainable_named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = config.init_obj('optimizer', module_optim, optimizer_grouped_parameters)

    lr_scheduler = config.init_obj('lr_scheduler', module_optim, optimizer)

    trainer = Trainer(model, metrics, optimizer,
                      config=config,
                      train_dataset=train_dataset,
                      valid_kwdlc_dataset=valid_kwdlc_dataset,
                      valid_kc_dataset=valid_kc_dataset,
                      valid_commonsense_dataset=valid_commonsense_dataset,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default='', type=str,
                        help='indices of GPUs to enable (default: "")')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for initialization')
    parsed_args = parser.parse_args()
    main(ConfigParser.from_args(parsed_args), parsed_args)
