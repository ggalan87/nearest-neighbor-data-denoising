# Pytorch lightning
This module is related to stuff that use [Pytorch Lightning](https://www.pytorchlightning.ai/) for training and/or inference.

## Pipelines
A basic notion of this module is the pipelines. Pipelines consist of standardized processes/recipes that are independent of datasets and models, and main input of them are a model and a dataset (actually a pytorch lightning datamodule). All pipelines are defined in [pipelines.py](pipelines/pipelines.py) file. The following pipelines are currently implemented.

### training
This performs a standard training procedure based on given arguments. Optionally evaluation is called right after training.

### testing
This performs inference based on an already trained model. More specifically it supports feature extraction (and save), features visualization by computing 2D UMAP embeddings and evaluation. The type of the evaluation is specified by argument and is one of:

| method | description |
| :--- | :--- |
| accuracy | The standard classification accuracy |
| reid | Ranking scores (CMC curve) and mAP computed using the raw features. Trainval samples are used for 'gallery' and test samples for query. |
| clustering | Various clustering scores computed on KMeans clustering that is given the ground-truth number of clusters. Non-standard evaluation, but it should give extra indications about the consistency of class samples if they are appropriately projected to nearby locations in the embedding space. |

The same pipeline supports also 

### explain
This utilizes the [captum](https://captum.ai) library and passes test samples into model interpreter (defined in the visualizations package) in order to visualize the impact of the input pixels to the predicted class. Currently, the algorithm that is used is hardcoded (IntegratedGradients). Some more effort is needed for visualizing impact of intermediate network parts etc.

### model visualization
This utilizes [torchviz](https://github.com/szagoruyko/pytorchviz) library in order to visualize the model architecture.

### aligned visualization
This is the only (currently) pipeline that expects a list of models, actually paths to models of the same architecture (model class) and computes AlignedUMAP embeddings simultaneously from all the models. Currently, part from which to compute the models (trainval or test) is hardcoded.

## Rest files
| filename | description                                                                                                                                                                       |
| :--- |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [data_modules.py](data/data_modules.py) | Here I define default transforms for some datasets and patch_visiondatamodule functionality which permits training using alternative samplers (needed for triplet-loss training). |
|[default_settings.py](cli_pipelines/default_settings.py)| Some default training hyperparams                                                                                                                                                 |
| [losses_playground.py](losses_playground.py)| Definition of PopulationBasedTripletLoss variant and other possible losses. There are losses that are not considered stable yet.                                                  |
|[models.py](models/models.py)| Models definitions                                                                                                                                                                |
|[samplers.py](data/samplers.py)| Samplers definitions                                                                                                                                                              |
| [train_cifar.py](train_cifar.py), [train_mnist.py](train_cifar.py) | Scripts which utilize the pipelines and the above clases/functions to train/test/visualize                                                                                        |
|[train_mnist_bolts.py](train_mnist_bolts.py)| Sample trainer/tester using pl_bolts API. Left behind for ideas on how to build custom models/pipelines and passing arguments                                                     |