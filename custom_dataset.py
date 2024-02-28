import mmcv
import numpy as np
from pathlib import Path
import os
from mmcls.datasets.builder import DATASETS
from mmcls.datasets.base_dataset import BaseDataset


@DATASETS.register_module()
class CustomDataset(BaseDataset):
    def __init__(self,
                 data_prefix,
                 pipeline,
                 classes=None,
                 ann_file=None,
                 test_mode=False):
        super().__init__(data_prefix, pipeline, classes, ann_file, test_mode)
        self.super_classes = [[1, 7], [2, 8], [3, 9], [4, 10], [5, 11], [6, 12], [13, 14]]

    def load_annotations(self):
        # Infer annotations from filenames, therefore annotation file is not needed
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
                data_infos.append(info)
            return data_infos


@DATASETS.register_module()
class OxfordCatsDogsBreeds(BaseDataset):
    def __init__(self,
                 data_prefix,
                 pipeline,
                 classes=None,
                 ann_file=None,
                 test_mode=False):
        super().__init__(data_prefix, pipeline, classes, ann_file, test_mode)

        self.super_classes = [[], []]
        self.load_class_info(data_prefix)
        print(self.super_classes)

    def load_class_info(self, data_prefix):
        """
        Parses a file of the form:
        #Image CLASS-ID SPECIES BREED ID
        #ID: 1:37 Class ids
        #SPECIES: 1:Cat 2:Dog
        #BREED ID: 1-25:Cat 1:12:Dog
        #All images with 1st letter as captial are cat images
        #images with small first letter are dog images
        <> <> <> <>
        ...
        <> <> <> <>
        :param data_prefix: The path to the root of the dataset
        :return: None
        """

        OxfordCatsDogsBreeds.CLASSES = []

        info_file = Path(data_prefix) / 'annotations/annotations/list.txt'
        infos = mmcv.list_from_file(info_file)[6:]
        for i in infos:
            file_name, class_id, species, _ = i.split(' ')
            class_name = file_name.rsplit('_', maxsplit=1)[0]

            if class_name not in OxfordCatsDogsBreeds.CLASSES:
                self.super_classes[int(species) - 1].append(int(class_id))
                OxfordCatsDogsBreeds.CLASSES.append(class_name)

    def load_annotations(self):
        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label, _, _ in samples:
                info = {
                    'img_prefix': os.path.join(self.data_prefix, 'images', 'images'),
                    'img_info': {'filename': filename + '.jpg'},
                    'gt_label_str': str(gt_label),
                    'gt_label': np.array(gt_label, dtype=np.int64)
                }
                data_infos.append(info)
            return data_infos
