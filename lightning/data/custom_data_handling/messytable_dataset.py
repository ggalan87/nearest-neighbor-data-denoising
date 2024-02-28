from pathlib import Path
import json
from pytorch_lightning import seed_everything
import functools
from torchvision.datasets import VisionDataset
from torchvision import transforms
from tqdm import tqdm
import copy
from dataclasses import dataclass
import cv2
from PIL import Image

# mine
from lightning.data.datamodules.identity_datamodule import IdentityDataModule
from lightning.data.data_modules import get_default_transforms
from lightning.data.samplers import RandomIdentitySampler
from lightning.data.dataset_filter import *

# Messy table dataset, different from ordinal classification or identity datasets, contains images with many instances
# inside. Therefore, a difference with them is in the _get_item_ method were the bounding box is needed such that we
# return the roi of the image which is relevant to the ID
# Unfortunately original Dataset class from MessyTable is a mess. __get_item__ from this does not return a single item
# rather than many items. It actually breaks the scope of __get_item__  as shown in:
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#getitem


@dataclass
class MessyTableDataEntry:
    """
    MessyTable dataset paper refers to 120 classes which correspond to 13 groups. Labeling however includes an
    intermediate grouping, where there are 42 unique class_ids, and 120 unique subclass_ids. I keep the naming scheme
    of the label file, and manually add the group label on top of them.

    @todo Consider converting labels to vector of labels according to the hierarchy levels as implemented in
      https://kevinmusgrave.github.io/pytorch-metric-learning/testers/#basetester
    """
    image_name: str
    group_id: int
    class_id: int
    subclass_id: int
    camera_id: str
    bounding_box: list
    group_description: str


class MessyTableDatasetPart(VisionDataset):
    available_data_parts = \
        [
            'test',
            'test_easy',
            'test_hard',
            'test_medium',
            'train',
            'val',
            'val_easy',
            'val_hard',
            'val_medium'
        ]

    def __init__(self,
                 dataset_root: str,
                 part_name: str,
                 dataset_filter: Optional[List[FilterBase]],
                 transforms: Optional[Callable] = None):
        super(MessyTableDatasetPart, self).__init__(root=dataset_root, transforms=transforms)

        dataset_root_path = Path(dataset_root)
        assert dataset_root_path.exists()
        assert part_name in self.available_data_parts

        self.part_name = part_name

        part_labels_path = dataset_root_path / 'labels' / f'{self.part_name}.json'
        assert part_labels_path.exists()

        self.data = self.process_labels(part_labels_path)

        if dataset_filter is not None:
            self._filter_by(dataset_filter)

        self.images_dir = dataset_root_path / 'images'

        # Find the images size by loading the first image / all images are the same size. The expected form is in
        # (height, width) format / kind of misleading for image size, rather more suitable for matrix / tensor size.
        # e.g. PIL.Image().size is in (width, height) format. That's why I separately get the required fields.
        sample_img = pil_loader(self.images_dir / self.data[0]['image_name'])
        self.img_size = (sample_img.height, sample_img.width)

    def __getitem__(self, index: int):
        # Get a copy of the entry, such that the original data are not affected below / need to keep only shallow info
        # and not the actual image data. For now, we do shallow copy because we expect that none of the data are nested.
        data_entry = copy.copy(self.data[index])

        image_path = self.images_dir / data_entry['image_name']

        scene_image = cv2.imread(str(image_path))
        image = self.crop_feat(scene_image, data_entry['bounding_box'], zoomout_ratio=1.0)
        image = Image.fromarray(image)

        if self.transforms is not None:
            # TODO: decide how to pass the target (label) for possible "relabel transform"
            image = self.transforms(image)
        else:
            bare_minimum_transform = transforms.ToTensor()
            image = bare_minimum_transform(image)

        data_entry['image'] = image

        # Not needed afterwards / we already got the crop we need, avoid increased memory demands
        # TODO: possibly put it as parameter, maybe needed for other reasons, e.g. simply pass the dataloader
        del data_entry['bounding_box']
        del data_entry['image_name']

        return data_entry

    def __len__(self):
        return len(self.data)

    def _filter_by(self, conditions_list: List[FilterBase]):
        filter_manager = FilterManager(conditions_list)
        self.data = filter_manager.apply_filters(self.data)

    @staticmethod
    def subcls_to_group_dict():
        groups = \
            [['A', (1, 10), 'bottled drinks'],
             ['B', (11, 19), 'cupped food'],
             ['C', (20, 30), 'canned food'],
             ['D', (31, 41), 'boxed food'],
             ['E', (42, 50), 'vacuum-packed food'],
             ['F', (51, 60), 'puffed food'],
             ['G', (61, 77), 'fruits'],
             ['H', (78, 83), 'vegetables'],
             ['I', (84, 96), 'staples'],
             ['J', (97, 100), 'utensils'],
             ['K', (101, 107), 'bowls & plates'],
             ['L', (108, 115), 'cups'],
             ['M', (116, 120), 'drink glasses']]

        subcls_to_group = {}
        subcls_to_group_description = {}
        for (group, subclasses, description) in groups:
            for c in range(subclasses[0], subclasses[1] + 1):
                subcls_to_group[c] = ord(group) - ord('A') + 1  # also convert letter to number, starting from 1
                subcls_to_group_description[c] = description
        return subcls_to_group, subcls_to_group_description

    def process_labels(self, path: Path):
        with open(path, 'r') as file:
            labels_content = json.load(file)

        # The other key is 'intrinsics', not parsed for now
        scenes_dict = labels_content['scenes']

        raw_data = []

        unique_cls = set()
        unique_subcls = set()
        unique_groups = set()

        subcls_to_group, subcls_to_group_description = self.subcls_to_group_dict()

        for scene_name, scene_info in tqdm(scenes_dict.items()):
            instances_summary = scene_info['instance_summary']
            cameras_info = scene_info['cameras']
            for cid, cameras_info in scene_info['cameras'].items():
                pathname = cameras_info['pathname']
                # ignore extrincs, corners
                instances = cameras_info['instances']
                for instance_id, instance_info in instances.items():
                    # Below we subtract by 1 all ids in order to be in range 0...total_ids-1.
                    # This is required at least by pytorch cross entropy loss but also useful for indexing in general
                    data_entry = \
                        MessyTableDataEntry(
                            image_name=pathname,
                            group_id=subcls_to_group[instance_info['subcls']] - 1,
                            class_id=instance_info['cls'] - 1,
                            subclass_id=instance_info['subcls'] - 1,
                            camera_id=cid,
                            bounding_box=instance_info['pos'],
                            group_description=subcls_to_group_description[instance_info['subcls']]
                        )
                    # Below I use __dict__ and not asdict method because it is implemented for recursive dictionaries
                    # and therefore slow. Here I don't want it
                    raw_data.append(data_entry.__dict__)

                    unique_cls.add(data_entry.class_id)
                    unique_subcls.add(data_entry.subclass_id)
                    unique_groups.add(data_entry.group_id)

        return raw_data

    @staticmethod
    def scalar_clip(x, min, max):
        """
        input: scalar
        """
        if x < min:
            return min
        if x > max:
            return max
        return x

    def crop_feat(self, img: np.ndarray, bbox: list, zoomout_ratio: float = 1.0):
        """
        input: img and requirement on zoomout ratio
        where img_size = (max_x, max_y)
        return: a single img crop
        """
        x1, y1, x2, y2 = bbox
        max_y, max_x = self.img_size

        # below clip is for MPII dataset, where bbox < 0
        x1 = self.scalar_clip(x1, 0, max_x)
        y1 = self.scalar_clip(y1, 0, max_y)
        x2 = self.scalar_clip(x2, 0, max_x)
        y2 = self.scalar_clip(y2, 0, max_y)

        img_feat = None

        if zoomout_ratio == 1.0:
            img_feat = img[int(y1):int(y2 + 1), int(x1):int(x2 + 1), :]
        elif zoomout_ratio > 1:
            h = y2 - y1
            w = x2 - x1
            img_feat = img[int(max(0, y1 - h * (zoomout_ratio - 1) / 2)):int(
                min(max_y - 1, y2 + 1 + h * (zoomout_ratio - 1) / 2)),
                       int(max(0, x1 - w * (zoomout_ratio - 1) / 2)):int(
                           min(max_x - 1, x2 + 1 + w * (zoomout_ratio - 1) / 2)), :]
        return img_feat


class MessyTableDatamodule(IdentityDataModule):
    """
    For now, I implement this data module as a subclass of identity data module because I want to act i =n this way.

    I utilize it as such:
    train: fit/train
    val: gallery
    test: query

    MessyTable dataset also includes easy, medium, hard subsets of the whole val and test datasets, which can be found
    in corresponding json files, e.g. test_easy.json, val_hard.json etc

    Also, the object is initialized with my flavour of transforms, i.e. dict transforms having keys the parts transforms,
    as also done in my previous implementations, e.g. for Market1501DatasetPart. Recently the pl_bolts introduced another
    approach, each transform to be separate as shown in: https://github.com/Lightning-AI/lightning-bolts/blob/7abe88a1c0d95660c0dc77adbf840dfaeab7c22a/pl_bolts/datamodules/vision_datamodule.py#L30

    TODO: Investigate if I will keep mine or theirs. Probably I prefer mine because it is not dedicated to scheme train, val, test
    """
    name = "messytable"
    dataset_cls = MessyTableDatasetPart
    dims = (3, 1080, 1920)

    def __init__(self,
                 data_dir: Optional[str] = None,
                 num_workers: int = 0,
                 batch_size: int = 32,
                 seed: int = 42,
                 shuffle: bool = True,
                 pin_memory: bool = True,
                 drop_last: bool = False,
                 training_sampler_class: Optional[type] = None,
                 test_subset_name: Optional[str] = None,
                 dataset_filter: Optional[List[FilterBase]] = None,
                 *args: Any,
                 **kwargs: Any,
                 ) -> None:
        super().__init__(data_dir=data_dir, num_workers=num_workers, batch_size=batch_size, seed=seed, shuffle=shuffle,
                         pin_memory=pin_memory, drop_last=drop_last, training_sampler_class=training_sampler_class,
                         *args, **kwargs)

        if test_subset_name is not None:
            assert test_subset_name in ['easy', 'medium', 'hard']
        self.test_subset_name = test_subset_name

        # Below I bind the dataset_filter parameter such that it is utilized for all instantiations afterwards
        self.dataset_cls = functools.partial(self.dataset_cls, dataset_filter=dataset_filter)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Creates train, val, and test dataset.

        Currently, val is not implemented, test for identity datasets consists of two parts, gallery and query
        """

        if stage == "fit" or stage is None:
            train_transforms = self.default_transforms() if self._train_transforms is None else self._train_transforms

            self.dataset_train = self.dataset_cls(
                self.data_dir, part_name='train', transforms=train_transforms)

        if stage == "test" or stage is None:
            part_suffix = f'_{self.test_subset_name}' if self.test_subset_name is not None else ''

            test_transforms = self.default_transforms() if self._test_transforms is None else self._test_transforms

            self.dataset_gallery = self.dataset_cls(
                self.data_dir, part_name='val'+part_suffix, transforms=test_transforms)
            self.dataset_query = self.dataset_cls(
                self.data_dir, part_name='test'+part_suffix, transforms=test_transforms)


if __name__ == '__main__':
    def dataset_part():
        dataset_filter = \
            [
                # LocalRandomKeepFilter(field_name='subclass_id', keep_probability=0.2),
                RegexListFilter(field_name='group_description', regex_values=[r'food']),
            ]
        dataset = MessyTableDatasetPart('/media/amidemo/Data/MessyTable', part_name='test_easy',
                                        dataset_filter=dataset_filter)

        # Pass one time all data to seek for errors
        for i in tqdm(range(len(dataset))):
            _ = dataset[i]

    def data_module():
        fn_keys = ('image', 'id', 'view_id')
        batch_unpack_fn = lambda batch_dict, keys=fn_keys: tuple(batch_dict[k] for k in keys)

        sampler_kwargs = \
            {
                'batch_size': 16,
                'num_instances': 4,
                'batch_unpack_fn': batch_unpack_fn,
                # Below we have to choose which of the labels will be used as id for creating the sampler
                'id_key': 'subclass_id',
            }

        dm = MessyTableDatamodule(data_dir='/media/amidemo/Data/MessyTable', test_subset_name='easy', batch_size=16,
                                  transforms=get_default_transforms('messytable'),
                                  training_sampler_class=RandomIdentitySampler, sampler_kwargs=sampler_kwargs,
                                  dataset_filter=[('subclass_id', range(0, 20))], num_workers=7)
        dm.setup('fit')
        ret = dm.train_dataloader()

        # Iterate dataloader
        for b in tqdm(ret):
            # print(b['subclass_id'])
            pass


    seed_everything(13)

    dataset_part()
    # data_module()
