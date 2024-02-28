from pytorch_lightning.callbacks import LearningRateMonitor
from lightning.ext import PeriodicCheckpoint
from pytorch_lightning import loggers as pl_loggers
from jsonargparse import lazy_instance
from lightning.data.datasets.noisy_mnist import obtain_noisy_label_configs
from run import CustomCLI

from pytorch_lightning.cli import ArgsType, LightningCLI


def cli_main(args: ArgsType = None):
    cli = CustomCLI(args=args)
    # ...
    pass


def batch_run_noisymnist():
    args = ['fit', '--config=configs/config_noisymnist.yaml']
    noisy_configs = obtain_noisy_label_configs(n_classes=10, n_excluded=(1, 2), noisiness_perc=1.0)

    for pa_option in [False, True]:
        for config in noisy_configs:
            config_args = args + \
                          [f'--data.dataset_args.init_args.labels_noise_perc={config}',
                           f'--model.init_args.with_population_aware={pa_option}']
            cli_main(config_args)


def batch_run_dirtymnist():
    args = ['fit', '--config=configs/config_dirtymnist.yaml']
    dirtiness_source_configs = ['self', 'emnist-l']

    for pa_option in [True]:
        for config in dirtiness_source_configs:
            config_args = args + \
                          [f'--data.dataset_args.init_args.dirtiness_source={config}',
                           f'--model.init_args.with_population_aware={pa_option}']
            cli_main(config_args)


if __name__ == "__main__":
   batch_run_dirtymnist()
