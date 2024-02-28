import sys
from typing import List, Tuple, Optional
from pathlib import Path
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


class Singleton (type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, every: int):
        super().__init__()
        self.every = every

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if pl_module.current_epoch % self.every == 0 or (pl_module.current_epoch + 1) == trainer.max_epochs:
            assert self.dirpath is not None
            current = Path(self.dirpath) / f"epoch-{pl_module.current_epoch}.ckpt"
            trainer.save_checkpoint(current)


class GradientMonitor(pl.Callback):
    def __init__(self, epochs_range: Optional[Tuple[int, int]] = None,
                 iterations_range: Optional[Tuple[int, int]] = None, parameters_to_log: Optional[List[str]] = None):
        super().__init__()

        if epochs_range is not None and iterations_range is not None:
            raise ValueError("Should have epochs range or iterations range, but not both")
        elif epochs_range is not None:
            self.epochs_range = range(*epochs_range)
            self.iterations_range = []
        elif iterations_range is not None:
            self.iterations_range = range(*iterations_range)
            self.epochs_range = []
        else:
            # Leave this case for intentionally muting the monitor. Can also be done using unused range, e.g. negative
            print(f'According to the options, {self.__class__.__name__} will be muted.')

        self.parameters_to_log = parameters_to_log

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.current_epoch in self.epochs_range or trainer.global_step in self.iterations_range:
            logger = trainer.logger

            if isinstance(logger, TensorBoardLogger):
                for name, param in trainer.model.backbone.named_parameters():
                    if self.parameters_to_log is not None and name not in self.parameters_to_log:
                        continue

                    if param.grad is not None:
                        logger.experiment.add_histogram(f'{name}.grad', param.grad, global_step=trainer.global_step)


class GlobalLogger(metaclass=Singleton):
    def __init__(self):
        self._logger = self._create_logger()

    def _create_logger(self):
        logger = logging.getLogger('lightning.pytorch')
        sh = logging.StreamHandler(sys.stdout)

        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%a, %d %b %Y %H:%M:%S')

        sh.setFormatter(formatter)
        logger.addHandler(sh)

        return logger

    @property
    def logger(self):
        return self._logger


logger = GlobalLogger().logger

