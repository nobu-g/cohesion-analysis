import math
from abc import abstractmethod

import torch
from numpy import inf

from utils import prepare_device
import data_loader.data_loaders as module_loader


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, metrics, optimizer, config, train_dataset):
        self.config = config
        cfg_trainer: dict = config['trainer']
        self.epochs: int = cfg_trainer['epochs']
        self.save_start_epoch: int = cfg_trainer.get('save_start_epoch', 1)
        self.logger = config.get_logger('trainer', cfg_trainer['verbosity'])

        # setup GPU device if available, move model into configured device
        self.device, device_ids = prepare_device(config['n_gpu'], self.logger)
        self.num_devices = max(len(device_ids), 1)
        self.model = model.to(self.device)
        if self.num_devices > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        max_bpg = self.config['trainer']['max_bpg']
        self.batches_per_optim = cfg_trainer['batch_size']
        self.gradient_accumulation_steps = math.ceil(self.batches_per_optim / (max_bpg * self.num_devices))
        batches_per_step = min(self.batches_per_optim, max_bpg * self.num_devices)
        if self.gradient_accumulation_steps > 1:
            self.config['data_loaders']['valid']['args']['batch_size'] = batches_per_step
        self.batches_per_device = math.ceil(batches_per_step / self.num_devices)
        self.config['data_loaders']['train']['args']['batch_size'] = batches_per_step
        self.data_loader = self.config.init_obj('data_loaders.train', module_loader, train_dataset)
        self.total_step = len(self.data_loader) * self.epochs
        self.optimization_step_per_epoch = math.ceil(len(self.data_loader) / self.gradient_accumulation_steps)
        self.total_optimization_step = self.optimization_step_per_epoch * self.epochs

        self.metrics = metrics
        self.optimizer = optimizer

        # configuration to monitor model performance and save best
        self.monitor: str = cfg_trainer.get('monitor', 'off')
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        self.logger.info("***** Running training *****")
        self.logger.info("  Num examples = %d", len(self.data_loader.dataset))
        self.logger.info("  Num Epochs = %d", self.epochs)
        self.logger.info("  Instantaneous batch size per device = %d", self.batches_per_device)
        self.logger.info("  Total train batch size (w. parallel & accumulation) = %d", self.batches_per_optim)
        self.logger.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)
        self.logger.info("  Total optimization steps = %d", self.total_optimization_step)
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged information into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged information to the screen
            for key, value in log.items():
                self.logger.info('{:40s}: {:.4f}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch >= self.save_start_epoch:
                self._save_checkpoint(epoch, save_best=best)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        save_path = self.config.save_dir / f'checkpoint-epoch{epoch}.pth'
        self.logger.info("Saving checkpoint: {} ...".format(save_path))
        torch.save(state, str(save_path))
        if save_best:
            best_path = self.config.save_dir / 'model_best.pth'
            self.logger.info("Saving current best: model_best.pth ...")
            if best_path.exists():
                best_path.unlink()
            best_path.resolve().symlink_to(save_path.name)

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
