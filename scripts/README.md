# scripts documentation
This folder contains a set of standalone scripts which enable executing some concepts of the repo

## Openset
[run_openset_pipeline.py](./run_openset_pipeline.py) runs an openset experiment, using OpensetTrainer api, given a set of features and algorithm parameters. The user has to specify the following:
1. Features as specified in [FeaturesStorage](../features_storage.py) format (see corresponding class)
2. Which classes to consider as known
3. Algorithm parameters, including openset algorithm and specific algorithm parameters

Given the above input the OpensetTrainer is able to train the model, make predictions and plots.