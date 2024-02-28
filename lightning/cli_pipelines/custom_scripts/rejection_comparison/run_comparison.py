import logging

import tqdm
from pytorch_lightning.utilities.seed import seed_everything
from lightning.ext import logger
#from lightning.cli_pipelines.custom_scripts.rejection_comparison.openset_based import run as openset_run
from lightning.cli_pipelines.custom_scripts.rejection_comparison.prism_based import run as prism_run
from lightning.cli_pipelines.custom_scripts.rejection_comparison.ssr_based import run as ssr_run
from lightning.cli_pipelines.custom_scripts.rejection_comparison.peach_based import run as peach_run
from inspect_openset_model_reduction import MetricsOverTimeLogger

seed_everything(13)
verbose = True

if verbose:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

# dataset_name, model_name, version, reference_epoch
options = \
    {
        'dataset_name': 'Food101N',
        'model_name': 'LitInception',
        'version': 1,
        'reference_epoch': 0,
        'batch_size': 64,
        'samples_fraction': 1.0
    }

#openset_run(**options)
# prism_run(**options)


metrics_logger = MetricsOverTimeLogger()
for e in tqdm.tqdm(range(0, 3)):
    options['reference_epoch'] = e
    # openset_run(**options)
    metrics_logger.add_metrics(prism_run(**options))

metrics_logger.export(f'prism_strategy.csv')

#prism_run(**options)
#prism_run(**options)
# ssr_run(**options)
# peach_run(**options)
