# Nearest Neighbor Data Denoising (N2D2)
UPDATE: Present repo is archived. The new repo containing refactored code and also additional implementations and experiments for person re-id can be found here: https://github.com/ggalan87/noise-robust-learning

This repository contains the code of the paper entitled "Nearest neighbor-based data denoising for deep metric learning" which was accepted in VISAPP 2024.

NOTE: The files of this repository have been imported from my internal repository and needs heavy cleanup of unneeded and deprecated files and few hardcoded paths. Possibly in the future I will cleanup and publish the original repo which also contains improvements and extensions and will archive this one.

The scripts which correspond to the experiments reported in the paper can be found in [experiments scripts folder](lightning/cli_pipelines/experiments_scripts). All experiments are mostly built around [lighning](./lightning) package which contains models, datasets etc. Data denoising methods can be found in [torch_metric_learning](./torch_metric_learning) package which is essentially an extension to [PML](https://github.com/KevinMusgrave/pytorch-metric-learning).

# Dependencies
The repository is mostly built around [pytorch lightning](https://github.com/Lightning-AI/pytorch-lightning) framework and [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning) library.

Parts of the repository includes code from other sources. Every such part has link to its source, e.g. url to the relevant implementation.

The [requirements](./requirements.txt) file includes the dependencies of the project, allthough some are deprecated, e.g. initial repo was built around mmcv and relevant packages which are not needed by the paper's code.

