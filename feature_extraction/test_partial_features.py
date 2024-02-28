from extract_features import extract
from pathlib import Path

"""
This is a script for testing features extraction for partially artificially occluded images with really occluded
counterparts
"""

# Options
# TODO: Move options to script argument
architecture = 'efficientnet_b3'
dataset_part = 'train'
datasets = [('OxfordCatsDogsBreeds', 'cats_n_dogs/Cats_and_Dogs_Breeds_Classification_Oxford_Dataset')]
dataset_root = '/media/amidemo/Data/'

imgs_root = Path('/home/amidemo/devel/workspace/object_classifier_deploy/visualizations/tmp_crops/')
image_names = \
    [
        #imgs_root / '20190925-00044-02-05-10.jpg',
        #imgs_root / '20190925-00044-03-05-10.jpg',
        #imgs_root / '20190925-00038-04-04-11.jpg'
        '20191208-00966-10-03-1.jpg',
        '20191001-00133-05-05-4.jpg',
        '20191208-00967-03-04-4.jpg',
        '20191208-00967-03-01-5.jpg',
        '20190925-00045-05-09-5.jpg',
        '20191208-00966-10-04-1.jpg',
        '20190925-00038-04-04-11.jpg'
    ]

image_paths = [imgs_root / img for img in image_names]
extract(architecture, dataset_part, datasets, dataset_root, image_paths)

