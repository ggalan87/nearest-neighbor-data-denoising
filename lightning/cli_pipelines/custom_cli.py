import importlib
from pytorch_lightning.cli import LightningCLI
import lightning.data.datasets
from lightning.data.data_modules import patch_visiondatamodule
from lightning.data.samplers import RandomIdentitySampler
from pytorch_lightning.callbacks import LearningRateMonitor
from lightning.ext import PeriodicCheckpoint
import torch.utils.data.sampler
import yaml
from pytorch_lightning import loggers as pl_loggers
from jsonargparse import lazy_instance
from utils import class_from_string


# def test_func(val: int):
#     print(f'TEST FUNC {val}')


class CustomCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # parser.add_subclass_arguments(RandomIdentitySampler, "sampler", fail_untyped=False, instantiate=False)
        parser.add_subclass_arguments(torch.utils.data.Sampler, "sampler", fail_untyped=False, instantiate=False)
        # parser.add_function_arguments(test_func, "test_func", fail_untyped=True)

        # Link number of classes from data to model
        parser.link_arguments("data.num_classes", "model.init_args.num_classes", apply_on="instantiate")
        parser.link_arguments("data.init_args.batch_size", "model.init_args.batch_size", apply_on="instantiate")

    def before_instantiate_classes(self) -> None:
        inner_config = self.config[self.subcommand] if self.subcommand is not None else self.config

        class_path = inner_config['sampler']['class_path']

        if 'RandomIdentitySampler' in class_path:
            patch_visiondatamodule(
                    sampler_class=class_from_string(class_path),
                    batch_size=inner_config['data']['init_args'].get('batch_size'),
                    num_instances=inner_config['sampler']['init_args'].get('num_instances'),
                    id_key=inner_config['sampler']['init_args'].get('id_key'),
                    fix_samples=inner_config['sampler']['init_args'].get('fix_samples'),
                    num_epochs=inner_config['trainer'].get('max_epochs'),
                    cached_oid_mapping=inner_config['sampler']['init_args'].get('cached_oid_mapping'),
            )
        else:
            patch_visiondatamodule(sampler_class=class_from_string(class_path))

        data_module_class = class_from_string(inner_config.data.class_path)

        # Override with <dataset_name>Ext variant whose __get_item__ method returns batch in dict instead of tuple if
        # the Ext class variant not already in the data module. This is the case for data modules defined in pl_bolts
        # dataset_cls_name = data_module_class.dataset_cls.__name__
        # if 'Ext' not in dataset_cls_name:
        #     data_module_class.dataset_cls = getattr(lightning.data.datasets, f'{dataset_cls_name}Ext')

        model_name = inner_config.model.class_path.rsplit('.', 1)[1]

        if self.subcommand == 'fit':
            # Construct experiment name based on datamodule name and model name
            experiment_name = f'{data_module_class.name}_{model_name}'
            inner_config.trainer.logger.init_args.name = experiment_name
