# Nearest Neighbor Data Denoising (N2D2)

This repository contains the code of the paper entitled "Nearest neighbor-based data denoising for deep metric learning" which was accepted in VISAPP 2024.

NOTE: The files of this repository have been imported from my internal repository and needs heavy cleanup of unneeded and deprecated files. Possibly in the future I will cleanup the original repo and will archive this one.

The scripts which correspond to the experiments reported in the paper can be found in [experiments scripts folder](lightning/cli_pipelines/experiments_scripts). All experiments and mostly built around [lighning](./lightning) package which contains models, datasets etc. Data denoising methods can be found in [torch_metric_learning](./torch_metric_learning) package which is essentially an extension to [PML](https://github.com/KevinMusgrave/pytorch-metric-learning).



