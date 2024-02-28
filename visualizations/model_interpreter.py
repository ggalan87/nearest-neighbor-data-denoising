import torch
from torchvision import transforms as T
import captum
from pytorch_lightning import LightningModule

from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from tqdm import tqdm

from matplotlib.colors import LinearSegmentedColormap


class ModelInterpreter:
    def __init__(self, model: LightningModule, image_normalization: T.Normalize = None):
        self._model = model

        # Change the model to evaluation mode
        self._model.eval()

        # Compute inverted normalization since the input image from the dataloader is already passed through transforms
        mean = torch.tensor(image_normalization.mean)
        std = torch.tensor(image_normalization.std)
        inverse_normalization = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

        # Empty transform in case of None
        self._inverse_normalization = inverse_normalization if inverse_normalization is not None else lambda x: x

    def interpret_input_impact(self, images: torch.Tensor, targets: torch.Tensor, algorithm_name: str,
                               visualization_sign='all', **kwargs):
        if algorithm_name not in captum.attr.__all__:
            raise RuntimeError(f'Algorithm name should be one of the supported:\n{captum.attr.__all__:}')

        images.requires_grad = True

        class_ = getattr(captum.attr, algorithm_name)
        algo = class_(self._model)

        ret = algo.attribute(images, target=targets, **kwargs)
        if isinstance(ret, tuple):
            attr = ret[0]
        else:
            attr = ret

        return self._visualize_images(self._inverse_normalization(images), attr,
                                      signs=['all', visualization_sign], title=algorithm_name)

    def interpret_layer_impact(self, images: torch.Tensor, targets: torch.Tensor, algorithm_name: str,
                               visualization_sign='all', **kwargs):
        raise NotImplementedError()

    def interpret_neuron_impact(self, images: torch.Tensor, targets: torch.Tensor, algorithm_name: str,
                               visualization_sign='all', **kwargs):
        raise NotImplementedError()

    def _visualize_images(self, orig_images: torch.Tensor, attributes, title='Input Importances', signs=['all', 'all'],
                          cmap='viridis', **kwargs):
        if len(orig_images.shape) != 4:
            raise RuntimeError('Input tensor should be in NCHW format (batch size, channels, height width)')

        default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                         [(0, '#ffffff'),
                                                          (0.25, '#000000'),
                                                          (1, '#000000')], N=256)

        figs = []
        for orig_image, attribute in tqdm(zip(orig_images, attributes), total=len(orig_images)):
            # Reorder from C, H, W to H, W, C and convert to numpy
            orig_image = orig_image.permute(1, 2, 0).detach().numpy()
            attribute = attribute.permute(1, 2, 0).detach().numpy()

            fig, ax = viz.visualize_image_attr_multiple(attribute, orig_image,
                                                        methods=['original_image', 'heat_map'],
                                                        signs=signs,
                                                        titles=['Original Image', title],
                                                        use_pyplot=False,
                                                        show_colorbar=True,
                                                        #cmap=cmap,
                                                        **kwargs)
            figs.append(fig)

        return figs
