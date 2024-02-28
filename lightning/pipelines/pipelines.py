import re
from time import time
from pathlib import Path
from typing import Union, Tuple, List, Type, Literal, Dict
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision import transforms as T

import features_storage
from features_inspector import FeaturesInspector
from visualizations.embeddings_visualization import EmbeddingsVisualizer
from visualizations.dataset_visualizer import DatasetVisualizer
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from visualizations import model_interpreter, model_visualizer
# from torchreid.metrics.rank import evaluate_rank
# from torchreid.metrics.distance import compute_distance_matrix

from .utils import get_features, get_parts_features
from lightning.ext import PeriodicCheckpoint


def training_pipeline(model: LightningModule, data_module: VisionDataModule, resume_from=None, evaluate=False, **kwargs):
    logging_path = f'./lightning_logs/'

    # Construct experiment name based on datamodule name and model name
    experiment_name = f'{data_module.name}_{type(model).__name__}'

    # Init trainer
    trainer = Trainer(
        logger=pl_loggers.TensorBoardLogger(logging_path, name=experiment_name),
        callbacks=[LearningRateMonitor(logging_interval="step"), PeriodicCheckpoint(1)],
        **kwargs
    )

    # Pass the datamodule as arg to trainer.fit to override model hooks :)
    trainer.fit(model, data_module, ckpt_path=resume_from)

    if evaluate:
        data_module.setup('test')
        trainer.test(model=model, datamodule=data_module)


def bench_k_means(kmeans, name, data, labels):
    """Benchmark to evaluate the KMeans initialization methods.
    Obtained from: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=None,  # was 300
        )
    ]

    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))


def testing_pipeline(data_module: VisionDataModule, model: LightningModule, extract_features=False,
                     visualize=False, evaluate=False, eval_type: Literal['accuracy', 'reid', 'clustering'] = 'accuracy',
                     feat_parts=None, batch_keys=None, output_tag='', output_path=None, images_server=None, gpus=None,
                     features_path: Path = None):
    # Change the model to evaluation mode
    model.eval()

    feats_storage = None

    if extract_features and features_path is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        save_path = features_path / f'{output_tag}.pt'

        if not Path(save_path).exists():
            feats_storage = get_parts_features(model, data_module, parts=feat_parts, batch_keys=batch_keys)
            feats_storage.save(save_path)
        else:
            print('Loaded cached features')
            feats_storage = features_storage.FeaturesStorage(cached_path=save_path)

        if visualize:
            visualization_pipeline(data_module, model, feats_storage,
                                   feat_parts=feat_parts, output_path=output_path, images_server=images_server)

    if evaluate:
        if eval_type == 'accuracy':
            data_module.setup('test')
            # Init trainer. In this case we omit logging for not creating a dummy version folder
            trainer = Trainer(
                logger=False,
                accelerator='gpu',
                devices=gpus
            )


            trainer.test(model=model, datamodule=data_module)

        elif eval_type == 'reid':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)

            if feats_storage is None:
                feats_storage = get_parts_features(model, data_module, parts=feat_parts, batch_keys=batch_keys)

            (trainval_feats, test_feats), (trainval_labels, test_labels) = feats_storage.raw_features()


            print('Computing distance matrix on feats...', trainval_feats.shape, test_feats.shape)
            dists_np = compute_distance_matrix(test_feats.to(device).half(), trainval_feats.to(device).half()).detach().cpu().numpy()
            test_labels_np = test_labels.detach().cpu().numpy().ravel()
            trainval_labels_np = trainval_labels.detach().cpu().numpy().ravel()
            print('Running evaluation...')
            all_cmc, mAP = evaluate_rank(dists_np, test_labels_np, trainval_labels_np,
                                         np.ones_like(test_labels_np), np.zeros_like(trainval_labels_np), max_rank=50)
            print(all_cmc, mAP)
        elif eval_type == 'clustering':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)

            if feats_storage is None:
                print('Extracting testing features / used as query')
                feats_storage = get_parts_features(model, data_module, parts=('test', ), batch_keys=batch_keys)

            (trainval_feats, test_feats), (trainval_labels, test_labels) = feats_storage.raw_features()

            test_feats_np = test_feats.detach().cpu().numpy()
            test_labels_np = test_labels.detach().cpu().numpy().ravel()

            kmeans = KMeans(init="k-means++", n_clusters=data_module.num_classes, n_init=4, random_state=0)
            print(82 * "_")
            print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")
            bench_k_means(kmeans=kmeans, name="k-means++", data=test_feats_np, labels=test_labels_np)
        else:
            print('Unknown evaluation method')


def visualization_pipeline(data_module: VisionDataModule, feats_storage: features_storage.FeaturesStorage,
                           features_tag: str, feat_parts=None, output_path=None, images_server=None):
    datasets = {}

    if 'trainval' in feat_parts:
        data_module.setup(stage='fit')
        datasets['trainval'] = data_module.dataset_train.dataset
    if 'test' in feat_parts:
        data_module.setup(stage='test')
        datasets['test'] = data_module.dataset_test

    visualizer = EmbeddingsVisualizer(output_path if output_path else './embeddings', dataset_name=data_module.name)
    visualizer.add_features(features_tag, feats_storage, datasets=datasets)
    visualizer.plot('bokeh', images_server=images_server)


def aligned_visualization_pipeline(data_module: VisionDataModule, checkpoints_paths: List[Union[str, Path]],
                                   model_class: Type[LightningModule]):
    visualizer = EmbeddingsVisualizer('./embeddings', data_module.name)

    feat_storage_objects = {}
    fi = FeaturesInspector()
    for i, cpath in enumerate(checkpoints_paths):
        if not Path(cpath).exists():
            raise FileNotFoundError(cpath)

        # Automatically find version from path
        try:
            version_number = re.search(r'version\_*([\d.]+)', cpath).group(1)
            model_version_string = f'version_{version_number}'
        except AttributeError:
            model_version_string = f'idx_{i}'

        # Load the model from the file and upload it to gpu
        model = model_class.load_from_checkpoint(cpath)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        feat_storage_object = get_parts_features(model, data_module, parts=['test'])
        feat_storage_objects[model_version_string] = feat_storage_object
        feats, labels = feat_storage_object.raw_features()
        fi.process('test', feats[1], labels[1])

    visualizer.add_features_multi(feat_storage_objects)
    visualizer.plot(backend='bokeh', single_figure=False)


def explain_pipeline(data: Union[VisionDataModule, Tuple[torch.Tensor, torch.Tensor, T.Normalize]],
                     model: LightningModule, show=False):
    """
    Produces some plots which show the focus of the model/layer/neuron on an input image. Algorithm is hardcoded for now.
    @param data:
    @param model:
    @param show:
    @return:
    """
    if isinstance(data, VisionDataModule):
        data_module = data

        # Ensure testing setup
        data_module.setup('test')

        mi = model_interpreter.ModelInterpreter(model=model, image_normalization=data_module.test_transforms.transforms[1])

        # Run the test data loader loader
        for batch_index, (images, labels) in enumerate(data_module.test_dataloader()):
            # pass the ground truth label in order to find which input parts are used to predict the correct label
            figs = mi.interpret_input_impact(images, targets=labels, algorithm_name='IntegratedGradients')
            for image_index, fig in enumerate(figs):
                fig.savefig(f'./plots/batch_{batch_index}_img_{image_index}.png')
            break
    else:
        image, label, normalization = data

        mi = model_interpreter.ModelInterpreter(model=model, image_normalization=normalization)

        # pass the ground truth label in order to find which input parts are used to predict the correct label
        # artificially add a dimension to represent batch index in the tensor format BCHW
        fig = mi.interpret_input_impact(image.unsqueeze(0), targets=label.unsqueeze(0), algorithm_name='IntegratedGradients')
        fig.savefig(f'./plots/sample.png')


def model_visualization_pipeline(model: LightningModule, data_module: VisionDataModule, **kwargs):
    """
    Visualization of the architecture of the model

    @param model: The model to be visualized
    @param data_module: A data_module for providing sample to the model
    @param kwargs:
    @return: None
    """
    mv = model_visualizer.ModelVisualizer('./plots')
    data_module.setup('test')

    # Loads the first sample from the data loader
    samples, labels = next(iter(data_module.test_dataloader()))
    mv.visualize(model, samples, **kwargs)


def dataset_visualization_pipeline(data_module: VisionDataModule):
    """
    Plot some images from each class of the dataset

    @param data_module: The data_module of the dataset
    @return: None
    """
    viz = DatasetVisualizer(1)
    data_module.setup('test')
    viz.visualize(data_module.dataset_test)

