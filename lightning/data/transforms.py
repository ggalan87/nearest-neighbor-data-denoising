from typing import Tuple
import random
import torch
import torchvision.transforms.functional as F


class RandomMinMaxTranslate(torch.nn.Module):
    """
    Randomly translates an image with magnitude in the range of (min, max) from the image center; values are expressed
    as proportions of each dimension. Original RandomAffine includes only max proportion and different max for each
    dimension (H, W). Instead, here I include same proportion.

    The original motivation of the transform instead of the available one is the following. We want to
    """
    def __init__(self, translation_limits: Tuple[float, float] = (0.25, 0.7)):
        super().__init__()
        self._translation_limits = translation_limits

    @staticmethod
    def random_sign():
        return 1 if random.random() < 0.5 else -1

    def forward(self, img):
        l_min, l_max = self._translation_limits
        _, h, w = img.shape

        min_dx, max_dx = l_min * w, l_max * w
        min_dy, max_dy = l_min * h, l_max * h

        # Implementation from RandfomAffine
        tx = int(round(torch.empty(1).uniform_(min_dx, max_dx).item()))
        ty = int(round(torch.empty(1).uniform_(min_dy, max_dy).item()))

        # My implementation
        # tx = random.randint(int(w * l_min), int(w * l_max))
        # ty = random.randint(int(h * l_min), int(h * l_max))

        x_sign = self.random_sign()
        y_sign = self.random_sign()

        # Below we utilize solely translation
        affine_params = \
            {
                'angle': 0,
                'translate': [tx * x_sign, ty * y_sign],
                'scale': 1.0,
                'shear': [0, 0]
            }

        return F.affine(img, **affine_params)
