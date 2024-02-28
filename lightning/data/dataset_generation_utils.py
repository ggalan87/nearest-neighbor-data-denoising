import itertools
from dataclasses import dataclass
from pathlib import Path
import shutil
import pickle
from typing import List

import yaml
import cv2
from numpy.random import default_rng
import numpy as np
import matplotlib
import torch
from pytorch_lightning.utilities.seed import seed_everything
from lightning.data.dataset_utils import random_split_perc, disturb_targets


@dataclass
class NormalDistribution:
    loc: float
    scale: float


class DatasetGenerator:
    configs = {}

    def __init__(self, root, overwrite, config_id):
        self.config_id = config_id
        self.config = self.configs[config_id]
        self._init_output_paths(root, overwrite)

    def _init_output_paths(self, root: Path, overwrite: bool):
        root.mkdir(exist_ok=True)
        self.config_path = root / 'config.yaml'
        if overwrite:
            # if not self.config_path.exists():
            #     raise AssertionError(f'Possible wrong root path {root}')
            # shutil.rmtree(root, ignore_errors=True)
            # root.mkdir(parents=True)
            pass
        elif len(list(Path(root).iterdir())) > 0:
            raise RuntimeError('Directory is not empty while overwrite is False!')

        # Create training and testing filepaths
        self.trainval_path = (root / 'trainval.pkl')
        self.gallery_path = (root / 'gallery.pkl')
        self.query_path = (root / 'query.pkl')

    def _save_dataset(self, trainval_data, gallery_data, query_data):
        with open(self.trainval_path, 'wb') as f:
            pickle.dump(trainval_data, f)

        with open(self.gallery_path, 'wb') as f:
            pickle.dump(gallery_data, f)

        with open(self.query_path, 'wb') as f:
            pickle.dump(query_data, f)

        with open(self.config_path, 'w') as f:
            _ = yaml.dump(self.config, f)

    def _split_test(self, test_data):
        gallery_indices, query_indices = random_split_perc(len(test_data['image']), self.config['gallery_ratio'])
        gallery_data = \
            {
                key: value[gallery_indices] if value is not None else None
                for (key, value) in test_data.items()
            }
        query_data = \
            {
                key: value[query_indices] if value is not None else None
                for (key, value) in test_data.items()
            }

        return gallery_data, query_data

    @staticmethod
    def generate_channel_samples(n_samples, img_h, img_w, rng, dists: List):
        def generate_channel(dist):
            vals = rng.normal(loc=dist.loc, scale=dist.scale, size=n_samples * img_h * img_w).reshape(
                (n_samples, img_h, img_w))
            vals = np.clip(vals, a_min=0.0, a_max=1.0)
            return vals

        channel_data = []
        for dist in dists:
            channel_data.append(generate_channel(dist))

        images = np.moveaxis(np.stack(channel_data), 0, -1)
        return images

    def generate_part_data(self, data_part):
        raise NotImplementedError

    def compute_mean_std(self, trainval_data):
        def mean_across_channel(data, channel):
            return round(float(torch.mean(data[:, :, :, channel] / 255.0).numpy()), 3)

        def std_across_channel(data, channel):
            return round(float(torch.std(data[:, :, :, channel] / 255.0).numpy()), 3)

        n_channels = trainval_data['image'].shape[3] if len(trainval_data['image'].shape) == 4 else 1
        if n_channels == 3:
            self.config['dataset_stats']['mean'] = \
                [
                    mean_across_channel(trainval_data['image'], 0),
                    mean_across_channel(trainval_data['image'], 1),
                    mean_across_channel(trainval_data['image'], 2),
                ]

            self.config['dataset_stats']['std'] = \
                [
                    std_across_channel(trainval_data['image'], 0),
                    std_across_channel(trainval_data['image'], 1),
                    std_across_channel(trainval_data['image'], 2),
                ]
        elif n_channels == 1:
            self.config['dataset_stats']['mean'] = \
                round(float(torch.mean(trainval_data['image'] / 255.0).numpy()), 3)
            self.config['dataset_stats']['std'] = \
                round(float(torch.std(trainval_data['image'] / 255.0).numpy()), 3)
        else:
            raise AssertionError('Only single channel or 3-channel images are supported')

    @staticmethod
    def get_means_positions(n, min_val=0, max_val=256):
        positions = []
        for i in range(n):
            mid_offset = (max_val - min_val) / (2 * n)
            segment_start = i * ((max_val - min_val) / n) + min_val
            pos = segment_start + mid_offset
            positions.append(pos)
        return positions

    def disturb_targets(self, part_data, targets, labels_noise_perc):
        part_data['target'], part_data['is_noisy'] = \
            disturb_targets(targets, labels_noise_perc)

        if part_data['target'] is None:
            part_data['target'] = targets
            part_data['is_noisy'] = torch.zeros_like(targets, dtype=torch.bool)

    def generate(self):
        seed_everything(13)

        trainval_data = self.generate_part_data('trainval')
        test_data = self.generate_part_data('test')

        gallery_data, query_data = self._split_test(test_data)

        self.compute_mean_std(trainval_data)

        self._save_dataset(trainval_data, gallery_data, query_data)


class GrayscaleNoiseDatasetGenerator(DatasetGenerator):
    configs = \
        {
            'v1': {
                'n_classes': 3,
                'trainval': {'samples_per_class': 2000, 'labels_noise_perc': {1: 1.0}},
                'test': {'samples_per_class': 1000, 'labels_noise_perc': {}},
                'gallery_ratio': 0.5,
                'image_size': {'width': 28, 'height': 28},
                'norm_std': 4,
                'dataset_stats': {}
            },
            'v2': {
                'n_classes': 3,
                'trainval': {'samples_per_class': 2000, 'labels_noise_perc': {1: 1.0}},
                'test': {'samples_per_class': 1000, 'labels_noise_perc': {}},
                'gallery_ratio': 0.5,
                'image_size': {'width': 28, 'height': 28},
                'norm_std': 70,
                'dataset_stats': {}
            }
        }

    def generate_part_data(self, data_part):
        part_data = \
            {
                'image': [],
                'target': []
            }

        def generate_samples(config, space_segments, pos, data_part):
            img_h = config['image_size']['height']
            img_w = config['image_size']['width']
            n_samples = config[data_part]['samples_per_class']
            std = config['norm_std']

            images = np.random.normal(pos * 255 / space_segments, std, n_samples * img_h * img_w).astype(
                np.uint8).reshape((n_samples, img_h, img_w))
            return images

        # e.g. 4 segments -> 5 positions including zero and max number, we keep only the 3 middle positions
        # for data classes
        space_segments = self.config['n_classes'] + 1
        part_data['image'].append(generate_samples(self.config, data_part=data_part,
                                                   space_segments=space_segments, pos=1))
        part_data['image'].append(generate_samples(self.config, data_part=data_part,
                                                   space_segments=space_segments, pos=2))
        part_data['image'].append(generate_samples(self.config, data_part=data_part,
                                                   space_segments=space_segments, pos=3))
        part_data['image'] = torch.tensor(np.vstack(part_data['image']))

        targets_list = []
        n_samples = self.config[data_part]['samples_per_class']

        # Normal labelling
        for target in range(self.config['n_classes']):
            targets_list.append(np.full((n_samples,), target, dtype=np.uint8))

        targets = torch.tensor(np.hstack(targets_list))

        self.disturb_targets(part_data, targets, self.config[data_part]['labels_noise_perc'])

        return part_data


class HSVNoiseDatasetGenerator(DatasetGenerator):
    configs = \
        {
            'v1': {
                'n_classes': 7,
                'trainval': {'samples_per_class': 1000, 'labels_noise_perc': {6: 1.0}},
                'test': {'samples_per_class': 1000, 'labels_noise_perc': {}},
                'gallery_ratio': 0.5,
                'image_size': {'width': 28, 'height': 28},
                'noise_properties': {'hue_scale': 0.05, 'saturation_scale': 0.5,
                                     'saturation_loc_normal': 0.75, 'saturation_loc_noisy': 0.55},
                'dataset_stats': {}
            },
            'v2': {
                'n_classes': 7,
                'trainval': {'samples_per_class': 1000, 'labels_noise_perc': {6: 1.0}},
                'test': {'samples_per_class': 1000, 'labels_noise_perc': {}},
                'gallery_ratio': 0.5,
                'image_size': {'width': 28, 'height': 28},
                'noise_properties': {'hue_scale': 0.05, 'saturation_scale': 0.5,
                                     'saturation_loc_normal': 0.75, 'saturation_loc_noisy': 0.6},
                'dataset_stats': {}
            }
        }

    def __init__(self, root, overwrite, config_id):
        super().__init__(root, overwrite, config_id)

    def generate_part_data(self, data_part):
        part_data = {}

        rng = default_rng()

        img_h = self.config['image_size']['height']
        img_w = self.config['image_size']['width']
        n_samples = self.config[data_part]['samples_per_class']
        n_populations = self.config['n_classes'] - 1

        classes_samples = []
        targets = []

        for i in range(n_populations):
            pos = i + 1
            hue_dist = NormalDistribution(pos / (n_populations + 1), self.config['noise_properties']['hue_scale'])
            saturation_dist = NormalDistribution(self.config['noise_properties']['saturation_loc_normal'],
                                                 self.config['noise_properties']['saturation_scale'])
            value_dist = NormalDistribution(0.8, 0.0)
            images_hsv = self.generate_channel_samples(n_samples, img_h, img_w, rng,
                                                       [hue_dist, saturation_dist, value_dist])
            images = (matplotlib.colors.hsv_to_rgb(images_hsv) * 255).astype(np.uint8)
            classes_samples.append(images)
            targets.extend([i] * n_samples)

        noisy_id = n_populations
        n_samples = int(n_samples / n_populations)
        for pos in range(1, n_populations + 1):
            hue_dist = NormalDistribution(pos / (n_populations + 1), self.config['noise_properties']['hue_scale'])
            saturation_dist = NormalDistribution(self.config['noise_properties']['saturation_loc_noisy'],
                                                 self.config['noise_properties']['saturation_scale'])
            value_dist = NormalDistribution(0.8, 0.0)
            images_hsv = self.generate_channel_samples(n_samples, img_h, img_w, rng,
                                                       [hue_dist, saturation_dist, value_dist])
            images = (matplotlib.colors.hsv_to_rgb(images_hsv) * 255).astype(np.uint8)
            classes_samples.append(images)
            targets.extend([noisy_id] * n_samples)

        images = torch.tensor(np.concatenate(classes_samples))
        targets = torch.tensor(np.array(targets))

        self.disturb_targets(part_data, targets, self.config[data_part]['labels_noise_perc'])

        part_data['image'] = images
        return part_data


class RGBNoiseDatasetGenerator(DatasetGenerator):
    configs = \
        {
            'v1': {
                'n_classes': 9,
                'trainval': {'samples_per_class': 1000, 'labels_noise_perc': {4: 1.0}},
                'test': {'samples_per_class': 1000, 'labels_noise_perc': {}},
                'gallery_ratio': 0.5,
                'image_size': {'width': 28, 'height': 28},
                'noise_properties': {'red_scale': 0.2, 'green_scale': 0.2, 'blue_scale': 0.01, 'min_val': 0.0,
                                     'max_val': 0.3},
                'dataset_stats': {}
            },
            'v2': {
                'n_classes': 9,
                'trainval': {'samples_per_class': 1000, 'labels_noise_perc': {4: 1.0, 6: 1.0}},
                'test': {'samples_per_class': 1000, 'labels_noise_perc': {}},
                'gallery_ratio': 0.5,
                'image_size': {'width': 28, 'height': 28},
                'noise_properties': {'red_scale': 0.2, 'green_scale': 0.2, 'blue_scale': 0.01, 'min_val': 0.0,
                                     'max_val': 0.2},
                'dataset_stats': {}
            },
        }

    def __init__(self, root, overwrite, config_id):
        super().__init__(root, overwrite, config_id)

    def generate_part_data(self, data_part):
        part_data = {}

        rng = default_rng()

        img_h = self.config['image_size']['height']
        img_w = self.config['image_size']['width']
        n_classes = self.config['n_classes']
        n_samples = self.config[data_part]['samples_per_class']
        min_val = self.config['noise_properties']['min_val']
        max_val = self.config['noise_properties']['max_val']
        classes_samples = []
        targets = []

        n_r = n_g = int(np.ceil(np.sqrt(n_classes)))
        r_means = self.get_means_positions(n_r, min_val=min_val, max_val=max_val)
        g_means = self.get_means_positions(n_g, min_val=min_val, max_val=max_val)
        combos = list(itertools.product(r_means, g_means))

        for i in range(n_classes):
            r_mean, g_mean = combos[i]
            r_dist = NormalDistribution(r_mean, self.config['noise_properties']['red_scale'])
            g_dist = NormalDistribution(g_mean, self.config['noise_properties']['green_scale'])
            b_dist = NormalDistribution(0.5, i * self.config['noise_properties']['blue_scale'])
            images = (self.generate_channel_samples(n_samples, img_h, img_w, rng, [r_dist, g_dist, b_dist])
                      * 255).astype(np.uint8)
            classes_samples.append(images)
            targets.extend([i] * n_samples)

        images = torch.tensor(np.concatenate(classes_samples))
        targets = torch.tensor(np.array(targets))

        self.disturb_targets(part_data, targets, self.config[data_part]['labels_noise_perc'])

        part_data['image'] = images
        return part_data


class RGBGradientNoiseDatasetGenerator(DatasetGenerator):
    configs = \
        {
            'v1': {
                'n_classes': 9,
                'trainval': {'samples_per_class': 1000, 'labels_noise_perc': {4: 1.0}},
                'test': {'samples_per_class': 1000, 'labels_noise_perc': {}},
                'gallery_ratio': 0.5,
                'image_size': {'width': 28, 'height': 28},
                'noise_properties': {'red_scale': 0.05, 'green_scale': 0.05, 'blue_scale': 0.01, 'min_val': 0.0,
                                     'max_val': 0.4},
                'dataset_stats': {}
            }
        }

    def __init__(self, root, overwrite, config_id):
        super().__init__(root, overwrite, config_id)

    def generate_rgb_gradients(self, n_samples, rng, r_dist, g_dist, b_dist, rot_dist):
        def get_gradient_2d(start, stop, width, height, is_horizontal):
            if is_horizontal:
                return np.tile(np.linspace(start, stop, width), (height, 1))
            else:
                return np.tile(np.linspace(start, stop, height), (width, 1)).T

        def get_gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
            result = np.zeros((height, width, len(start_list)), dtype=float)

            for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
                result[:, :, i] = get_gradient_2d(start, stop, width, height, is_horizontal)

            return result

        r_vals = rng.normal(loc=r_dist.loc, scale=r_dist.scale, size=n_samples)
        r_vals = np.clip(r_vals, a_min=0.0, a_max=1.0)
        g_vals = rng.normal(loc=g_dist.loc, scale=g_dist.scale, size=n_samples)
        g_vals = np.clip(g_vals, a_min=0.0, a_max=1.0)
        b_vals = rng.normal(loc=b_dist.loc, scale=b_dist.scale, size=n_samples)
        b_vals = np.clip(b_vals, a_min=0.0, a_max=1.0)
        rotations = rng.normal(loc=rot_dist.loc, scale=rot_dist.scale, size=n_samples)
        rotations = np.clip(rotations, a_min=0.0, a_max=1.0)

        colors = np.moveaxis(np.stack((r_vals, g_vals, b_vals)), 0, -1)
        colors = (colors * 255).astype(np.uint8)
        rotations = (rotations * 360).astype(np.uint8)

        w = self.config['image_size']['width']
        h = self.config['image_size']['height']
        up_w = 2 * w
        up_h = 2 * h
        min_cut = 0.20
        max_cut = 0.80

        images = []
        for i in range(n_samples):
            image = np.uint8(get_gradient_3d(up_w, up_h, (0, 0, 0), colors[i], (False, False, False)))
            M = cv2.getRotationMatrix2D((up_w / 2, up_h / 2), rotations[i], 1.0)
            image = cv2.warpAffine(image, M, (up_w, up_w))
            image = image[int(min_cut * up_h):int(max_cut * up_h), int(min_cut * up_w):int(max_cut * up_w)]
            image = cv2.resize(image, (w, h))
            images.append(image)

        images = np.stack(images)
        return images

    def generate_part_data(self, data_part):
        part_data = {}

        rng = default_rng()

        img_h = self.config['image_size']['height']
        img_w = self.config['image_size']['width']
        n_classes = self.config['n_classes']
        n_samples = self.config[data_part]['samples_per_class']
        min_val = self.config['noise_properties']['min_val']
        max_val = self.config['noise_properties']['max_val']
        classes_samples = []
        targets = []

        n_r = n_g = int(np.ceil(np.sqrt(n_classes)))
        r_means = self.get_means_positions(n_r, min_val=min_val, max_val=max_val)
        g_means = self.get_means_positions(n_g, min_val=min_val, max_val=max_val)
        rot_means = self.get_means_positions(n_classes, min_val=0, max_val=1)
        combos = list(itertools.product(r_means, g_means))

        for c in range(n_classes):
            r_mean, g_mean = combos[c]
            r_dist = NormalDistribution(r_mean, self.config['noise_properties']['red_scale'])
            g_dist = NormalDistribution(g_mean, self.config['noise_properties']['green_scale'])
            b_dist = NormalDistribution(0.5, c * self.config['noise_properties']['blue_scale'])
            rot_dist = NormalDistribution(rot_means[c], 0.05)
            images = self.generate_rgb_gradients(n_samples, rng, r_dist, g_dist, b_dist, rot_dist)
            classes_samples.append(images)
            targets.extend([c] * n_samples)

        images = torch.tensor(np.concatenate(classes_samples))
        targets = torch.tensor(np.array(targets))

        self.disturb_targets(part_data, targets, self.config[data_part]['labels_noise_perc'])

        part_data['image'] = images
        return part_data

