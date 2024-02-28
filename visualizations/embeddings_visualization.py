import pickle
from typing import List, Dict, Optional, Union
from warnings import warn
from urllib.parse import urljoin
import numpy as np
import numpy as np
import sklearn.datasets
import torch
from sklearn.decomposition import PCA
import umap
import umap.plot
import umap.utils as utils
import umap.aligned_umap
import matplotlib.image
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.datasets import VisionDataset
import pandas as pd
from tqdm import tqdm
from .bokeh_plotter import BokehPlotter


from .umap_utils import create_identical_relation_dict_same

from features_storage import FeaturesStorage


class EmbeddingsVisualizer:
    def __init__(self, output_directory: str, dataset_name: Optional[str]):
        self._output_directory = Path(output_directory)
        self._dataset_name = dataset_name
        self._output_directory.mkdir(parents=True, exist_ok=True)
        self._available_backends = ['matplotlib', 'bokeh']

        self._embeddings = {}

    def add_features(self, features_id: str, features: FeaturesStorage, datasets: Optional[Dict[str, VisionDataset]]):
        """
        The purpose of this method is to store embeddings of the given features for later plotting. The high dimensional
         features are first projected into 2D embeddings and then stored in a dictionary

        @param features_id: A unique id which characterizes the added features, e.g. dataset name, model name etc
        This will be used in the title of  the plot during visualization
        @param features: A FeaturesStorage instance, which holds
        @param datasets: A dictionary which holds mappings of dataset_parts to VisionDataset instances from which the
        features were computed
        @return: None
        """
        embeddings, mapper = self._compute_embeddings(features, datasets)
        self._embeddings[features_id] = \
            {
                'embeddings': embeddings,
                'mapper': mapper
            }

    def add_features_multi(self, features_dict: Dict[str, FeaturesStorage], datasets: Optional[Dict[str, VisionDataset]]):
        """
        The purpose of this method is to store embeddings of the given features for later plotting. The difference to
        add_features is that multiple features sets obtained from the same dataset e.g. through different models are
        passed in order to simultaneously compute aligned embeddings, and therefore get better visualizations that can
        be directly compared. As in the plain case, the features are first projected into 2D embeddings and then stored
        in a dictionary

        @param features_dict: A dict which maps unique features id (e.g. model name, model variant, dataset name) to its
        corresponding FeaturesStorage instance
        @param datasets: A dictionary which holds mappings of dataset_parts to VisionDataset instances from which the
        features were computed
        @return: None
        """
        embeddings_list, mapper = self._compute_aligned_embeddings(list(features_dict.values()), datasets)

        for i, features_id in enumerate(features_dict.keys()):
            self._embeddings[features_id] = \
                {
                    'embeddings': embeddings_list[i],
                    'mapper': mapper
                }

        dataset_dir = (self._output_directory / self._dataset_name
                       if self._dataset_name is not None else '')
        dataset_dir.mkdir(exist_ok=True)
        output_filepath = dataset_dir / f'umap_aligned_embeddings.pkl'
        pickle.dump(embeddings_list, open(output_filepath, 'wb'))

    def _compute_embeddings(self, features: FeaturesStorage, datasets: Optional[Dict[str, VisionDataset]]):
        # Utilize a default UMAP instance
        # TODO: Refactor to support densmap or do not utilize the class for such specific thing at all
        mapper = umap.UMAP(verbose=True, densmap=False)

        embeddings = {}

        # Firstly fit the mapper to the best possible features set
        training_feats = features.training_feats
        testing_feats = features.testing_feats

        if training_feats is not None:
            mapper.fit(training_feats['feats'])
        elif testing_feats is not None:
            mapper.fit(testing_feats['feats'])
        else:
            raise RuntimeWarning('No available features for computing UMAP embeddings.')

        # Second apply transformation to all features
        for part, feat in features['feats'].items():
            embeddings[part] = \
                {
                    'data': mapper.transform(feat['feats'].numpy()),
                    'labels': feat[features.target_key].numpy()
                }

            # Also store additional info if available
            if datasets is not None and 'data_idx' in feat:
                embeddings[part]['dataset'] = datasets[part]
                embeddings[part]['data_idx'] = feat['data_idx'].numpy()

            if 'predicted_target' in feat:
                embeddings[part]['predicted_labels'] = embeddings[part]

            manually_added = ['feats', features.target_key, 'data_idx', 'predicted_target']
            for k, v in feat.items():
                if k in manually_added:
                    continue
                if isinstance(v, torch.Tensor):
                    v = v.numpy()
                embeddings[part][k] = v

        return embeddings, mapper

    def _compute_aligned_embeddings(self, features_list: List[FeaturesStorage],
                                    datasets: Optional[Dict[str, VisionDataset]]):
        # Integrity check
        def input_is_valid(training_feats_l, testing_feats_l):
            def check(l):
                # at least two elements and either all none or all of equal size (if not None, handled by short
                # circuit evaluation)
                return len(l) > 1 and (all(list(map(lambda x: x is None, l))) or all(
                    [len(l[0]['feats']) == len(elem['feats']) for elem in l[1:]]))

            # Check individually and afterwards check that they are not simultaneously None
            return check(training_feats_l) and check(testing_feats_l) and not all(
                elem is None for elem in [training_feats_l[0], testing_feats_l[0]])

        training_feats_list = [fs.training_feats for fs in features_list]
        testing_feats_list = [fs.testing_feats for fs in features_list]
        if not input_is_valid(training_feats_list, testing_feats_list):
            raise AssertionError('features_list should contain at least two elements of type FeaturesStorage and each '
                                 'element should contain either training or testing features (or both)')

        # Decide from which features to create the mapping, priority: (1) trainval, (2) test
        if training_feats_list[0] is not None:
            fitting_features_list = training_feats_list
            part = 'trainval' if 'trainval' in features_list[0].get_parts_tags() else 'train'
        else:
            fitting_features_list = testing_feats_list
            part = 'test'

        # Keep raw feats without the labels
        raw_features_list = [feat['feats'] for feat in fitting_features_list]

        constant_relations = create_identical_relation_dict_same(raw_features_list[0].shape[0],
                                                                 n_feats=len(raw_features_list))

        # Leave default neighbors for now
        aligned_mapper = umap.AlignedUMAP(verbose=True).fit(raw_features_list, y=None,  # n_neighbors=100,
                                                            relations=constant_relations)

        embeddings_list = []

        for i, model_feats in enumerate(features_list):
            embeddings = {}
            storage_object = features_list[i]
            embeddings[part] = \
                {
                    'data': aligned_mapper.embeddings_[i],
                    'labels': storage_object['feats'][part][storage_object.target_key].numpy()
                }

            # Also store additional info if available
            if datasets is not None and 'data_idx' in storage_object['feats'][part]:
                embeddings[part]['dataset'] = datasets[part]

                # Distinguish case when the feature storage is filtered
                if storage_object.kept_indices is None:
                    # Normal case / not filtered
                    embeddings[part]['data_idx'] = storage_object['feats'][part]['data_idx'].numpy()
                else:
                    embeddings[part]['data_idx'] = \
                        storage_object['feats'][part]['data_idx'][storage_object.kept_indices[part]].numpy()

            embeddings_list.append(embeddings)

        return embeddings_list, aligned_mapper

    def _populate_images(self, dataset: VisionDataset, part: str, dataset_idx: np.ndarray, images_directory: str,
                         server_url: str) -> Optional[List[str]]:
        """
        Returns a list which contains images paths, suitable for embedding in a bokeh plot. We have two cases:
        (a) the dataset is in an in-memory format
        (b) the dataset is already saved in files

        Either case the images are created or symlinked in images_directory which is meant to be observed by an images
        server. As the images should be with unique filenames, we prepend the part name to the path

        @param dataset: The images dataset which will be used for the visualization
        @param part: Part name of the dataset, as the dataset should normally be part of a larger dataset, encapsulated
        in a data_module
        @param dataset_idx: The array which holds indices of the dataset. This is to associate features/labels data
        that were possibly created in random order by the sampler class
        @part images_directory: The directory in which images will be populated
        @return: A list of serverd paths of images, i.e. paths under server_url
        """

        if dataset is None or dataset_idx is None:
            warn('Images are not available for plotting')
            return None

        # Decide the case based on the first idx
        zero_idx = dataset_idx[0]
        dataset_entry = dataset[zero_idx]
        image_file_is_available = True if 'image_path' in dataset_entry else False

        if image_file_is_available:
            image_server_paths = self._link_images(dataset=dataset, part=part, dataset_idx=dataset_idx,
                                                   images_directory=images_directory, server_url=server_url)
        else:
            image_server_paths = self._save_images(dataset=dataset, part=part, dataset_idx=dataset_idx,
                                                   images_directory=images_directory, server_url=server_url,
                                                   overwrite=False)

        return image_server_paths

    def _local_to_server_path(self, local_path: str, server_url: str) -> str:
        local_path = local_path.replace(str(self._output_directory), '.')
        return urljoin(server_url, local_path)

    def _link_images(self, dataset: VisionDataset, part: str, dataset_idx: np.ndarray, images_directory: str,
                         server_url: str):

        image_server_paths = []

        for idx in dataset_idx:
            # 0. Get the path
            img_path = Path(dataset.data[idx]['image_path'])
            # The path above is in absolute form, we need to symlink the image to the specified images_directory
            # and then append it to the list

            # 1. Create new image name as <part_name>_<idx>_orig_img_name
            img_name = f'{part}_{idx}_{img_path.name}'

            # 2. Create the new path
            link_path = Path(images_directory) / img_name

            # 3. Create the link to the original path
            try:
                link_path.symlink_to(img_path)
            except FileExistsError:
                pass

            # 4. Construct server path
            server_path = self._local_to_server_path(local_path=str(link_path), server_url=server_url)

            # 4. Append path to images
            image_server_paths.append(server_path)

        return image_server_paths

    def _save_images(self, dataset: VisionDataset, part: str, dataset_idx: np.ndarray, images_directory: str,
                         server_url: str, overwrite=False) -> List[str]:
        # If images are in ndarray form we have to save them to actual files in order to include them to the
        # visualization. This is not mandatory for smaller datasets but for now I implement only this option in
        # order to support larger datasets and "lazy loading" of the image rather than in memory
        # TODO: support in-memory images for smaller datasets

        def get_image_from_record(dataset: VisionDataset, idx):
            img = dataset.data[idx]['image']
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            return img

        def get_raw_image(dataset: VisionDataset, idx):
            img = dataset.data[idx]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            return img

        if isinstance(dataset.data, list) and isinstance(dataset.data[0], dict) and \
            'image' in dataset.data[0] and isinstance(dataset.data[0]['image'], (np.ndarray, torch.Tensor)):
            obtain_image_func = get_image_from_record
        elif isinstance(dataset.data, np.ndarray):
            obtain_image_func = get_raw_image
        else:
            raise NotImplementedError

        image_server_paths = []
        for idx in dataset_idx:
            img = obtain_image_func(dataset, idx)
            filepath = Path(images_directory) / f'{part}_{idx}.png'

            server_path = self._local_to_server_path(local_path=str(filepath), server_url=server_url)
            image_server_paths.append(server_path)

            if filepath.exists() and not overwrite:
                continue

            matplotlib.image.imsave(filepath, img)

        return image_server_paths

    def plot(self, backend: str, single_figure=False, images_server=None):
        if len(self._embeddings) == 0:
            warn(
                f'No embeddings have computed yet. Need to call {self.add_features.__name__} or '
                f'{self.add_features_multi.__name__} at least once.')
            return

        if backend not in self._available_backends:
            raise ValueError(f'Invalid plotting backend {backend}. Should be one of {self._available_backends}')

        def pass_embeddings():
            for features_id, embeddings_dict in self._embeddings.items():
                for part, embeddings_info in embeddings_dict['embeddings'].items():
                    yield features_id, part, embeddings_info

        if backend == 'matplotlib':
            if single_figure:
                raise NotImplementedError
            else:
                for features_id, part, embeddings_info in pass_embeddings():
                    data = embeddings_info['data']
                    labels = embeddings_info['labels']

                    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
                    ax_bound = self.__class__.axis_bounds(np.vstack(data))

                    scatter = ax.scatter(*data.T, s=2, c=labels, cmap="Spectral", alpha=0.3)
                    ax.axis(ax_bound)
                    ax.set(xticks=[], yticks=[])
                    ax.set_title(f'{features_id} - {part}')
                    plt.tight_layout()

                    # If dataset name is given then append it to the output path
                    output_dir = self._output_directory / \
                        self._dataset_name if self._dataset_name is not None else ''

                    output_dir.mkdir(exist_ok=True)
                    #plt.savefig(output_dir / f'{features_id}_{part}.png')

                plt.show()
        elif backend == 'bokeh':
            for features_id, part, embeddings_info in pass_embeddings():
                data = embeddings_info['data']
                labels = embeddings_info['labels']
                dataset = embeddings_info.get('dataset')
                predicted_labels = embeddings_info.get('predicted_labels')
                noisy_samples = embeddings_info.get('is_noisy')

                dataset_dir = (self._output_directory / self._dataset_name
                               if self._dataset_name is not None else '')

                images_dir = dataset_dir / 'images'
                images_dir.mkdir(exist_ok=True, parents=True)

                # Images are required for the purpose of visualization
                image_server_paths = self._populate_images(dataset, part, embeddings_info.get('data_idx'),
                                                           images_dir, images_server)

                output_filepath = dataset_dir / f'umap_plot_{features_id}_{part}.html'

                plotter = BokehPlotter()
                plotter.plot(
                    plot_title=features_id,
                    embeddings_2d=data,
                    labels=labels,
                    names_dict=dataset.classes,
                    image_server_paths=image_server_paths,
                    output_filepath=output_filepath,
                    predicted_labels=predicted_labels,
                    noisy_samples=noisy_samples
                )

    def get_mapper(self, features_id):
        return self._embeddings[features_id]['mapper']

    @staticmethod
    def axis_bounds(embedding):
        """
        Computes bounds for axes to use them while plotting UMAP
        :param embedding: The embeddings from which to compute the bounds
        :return: the axes bounds
        """
        left, right = embedding.T[0].min(), embedding.T[0].max()
        bottom, top = embedding.T[1].min(), embedding.T[1].max()
        adj_h, adj_v = (right - left) * 0.1, (top - bottom) * 0.1
        return [left - adj_h, right + adj_h, bottom - adj_v, top + adj_v]


