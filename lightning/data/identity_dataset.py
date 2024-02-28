from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torchvision.datasets import DatasetFolder, ImageFolder, VisionDataset
from PIL import Image
from pathlib import Path
import glob
import os.path as osp
import re
import copy
from lightning.data.dataset_utils import pil_loader


class IdentityImageDataset(VisionDataset):
    """
    https://github.com/pytorch/vision/issues/215
    https://github.com/pytorch/vision/issues/230
    https://github.com/pytorch/vision/issues/5324

    I choose to use only the new API with transforms keyword
    """

    # A dict which maps dataset part names to paths relative to dataset root, e.g. 'train': <dataset_root>/train_dir
    parts_folders: Dict

    def __init__(
            self,
            dataset_root: str,
            part_name: str,
            transforms: Optional[Callable] = None,
            relabel=False
    ) -> None:
        super(IdentityImageDataset, self).__init__(dataset_root, transforms=transforms)
        self.part_name = part_name
        self.relabel = relabel

    # def __getitem__(self, index: int) -> Any:
    #     img_path, pid, camid, dsetid = self.data[index]
    #
    #     img = read_image(img_path)
    #     if self.transform is not None:
    #         img = self._transform_image(self.transform, self.k_tfm, img)
    #     item = {
    #         'img': img,
    #         'pid': pid,
    #         'camid': camid,
    #         'impath': img_path,
    #         'dsetid': dsetid
    #     }
    #
    #     return item

    def __len__(self) -> int:
        pass

    def get_part_directory(self) -> Path:
        if self.part_name not in self.parts_folders:
            raise ValueError(f'Folder name for part {self.part_name} is not provided.')

        part_directory = Path(self.root) / self.parts_folders[self.part_name]

        if not part_directory.exists():
            raise ValueError(f'Path for part {self.part_name} does not exist: {part_directory}')

        return part_directory

    @property
    def name(self):
        return self.part_name


class Market1501DatasetPart(IdentityImageDataset):
    parts_folders = \
        {
            'train': 'bounding_box_train',
            'gallery': 'bounding_box_test',
            'query': 'query'
        }

    def __init__(
            self,
            dataset_root: str,
            part_name: str,
            transforms: Optional[Callable] = None,
            relabel=False
    ):
        super(Market1501DatasetPart, self).__init__(dataset_root, part_name, transforms=transforms, relabel=relabel)
        self.data = self.process_dir(self.get_part_directory())

    def __getitem__(self, index: int) -> Any:
        # Get a copy of the entry, such that the original data are not affected below / need to keep only shallow info
        # and not the actual image data. For now, we do shallow copy because we expect that none of the data are nested.
        data_entry = copy.copy(self.data[index])

        image_path = data_entry['image_path']

        image = pil_loader(image_path)

        if self.transforms is not None:
            # TODO: decide how to pass the target (label) for possible "relabel transform"
            image = self.transforms(image)

        data_entry['image'] = image

        return data_entry

    def __len__(self) -> int:
        return len(self.data)

    def process_dir(self, dir_path) -> List[Dict]:
        print(f'Processing {dir_path}')
        pattern = re.compile(r'([-\d]+)_c(\d)')
        images_dir = Path(dir_path)

        img_paths = list(images_dir.glob('*.jpg'))
        pid_container = set()

        def extract_filename_fields(paths):
            for path in paths:
                pid, cid = map(int, pattern.search(str(path)).groups())
                if pid == -1:
                    continue  # junk images are just ignored
                yield path, pid, cid

        for img_path, pid, _ in extract_filename_fields(img_paths):
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for dataset_idx, (img_path, pid, cam_id) in enumerate(extract_filename_fields(img_paths)):
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= cam_id <= 6

            cam_id -= 1  # index starts from 0

            if self.relabel:
               pid = pid2label[pid]

            data_entry = \
                {
                    'image_path': str(img_path),
                    'id': pid,
                    'view_id': cam_id,
                    'data_idx': dataset_idx
                }

            data.append(data_entry)

        # Map pid to a corresponding name. This is the same with prefix, but keep it here for completeness /
        # compatibility with named datasets
        self.classes = {pid: f'ID_{pid}' for pid, _ in pid2label.items()}
        return data


if __name__ == '__main__':
    market_dir = Path('/media/amidemo/Data/reid_datasets/market1501/Market-1501-v15.09.15/')

    market_train = Market1501DatasetPart(str(market_dir), part_name='gallery', transforms=None, relabel=False)
    ids = set()
    for data_elem in market_train:
        ids.add(data_elem['id'])
