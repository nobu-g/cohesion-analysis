import os
import logging
from datetime import datetime
from functools import partial
from pathlib import Path

from logger import setup_logging
from utils import read_json, write_json

LOG_LEVELS = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG
}


class ConfigParser:
    def __init__(self, config, result_dir=None, resume=None, run_id=None):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules,
        checkpoint saving and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file
         for example.
        :param result_dir: path to base directory where checkpoints are saved.
        :param resume: String, path to the checkpoint being loaded.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log.
         Timestamp is being used as default
        """
        self._config = config
        self.resume = resume

        if self.config['trainer']['save_dir'] == '':
            return

        self.name = self.config['name']
        self.run_id = run_id
        if run_id is None:
            self.run_id = datetime.now().strftime(r'%m%d_%H%M%S')  # use timestamp as default run-id
        self.result_dir = result_dir or Path(self.config['trainer']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / 'config.json')

        # configure logging module
        setup_logging(self.log_dir, log_config='src/logger/logger_config.json')

    @classmethod
    def from_args(cls, args, run_id=None, inherit_save_dir=False):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        result_dir = resume = ensemble = None
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_file = resume.parent / 'config.json'
            if inherit_save_dir is True:
                result_dir = resume.parents[2]
                run_id = run_id or str(resume.parent.name)
        elif getattr(args, 'ens', None) is not None:
            cfg_file = next(Path(args.ens).glob('*/config.json'))
            ensemble = args.ens
            if run_id is None:
                run_id = ''
        elif args.config is not None:
            cfg_file = Path(args.config)
        else:
            raise ValueError("Configuration file need to be specified. Add '-c config.json', for example.")

        config = read_json(cfg_file)
        if args.config and (resume or ensemble):
            config.update(read_json(args.config))
            if resume is not None and config['name'] != str(resume.parent.parent.name):
                if run_id is None:
                    run_id = str(resume.parent.parent.name)

        return cls(config, result_dir=result_dir, resume=resume, run_id=run_id)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        cfg = self
        for sub in name.split('.'):
            cfg = cfg[sub]
        module_name = cfg['type']
        module_args = dict(cfg['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    @staticmethod
    def get_logger(name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, LOG_LEVELS.keys())
        assert verbosity in LOG_LEVELS, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(LOG_LEVELS[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self) -> dict:
        return self._config

    @property
    def expr_dir(self) -> Path:
        return self.result_dir / self.name

    @property
    def save_dir(self) -> Path:
        return self.result_dir / self.name / self.run_id

    @property
    def log_dir(self) -> Path:
        return self.save_dir
