from mmcv.image.photometric import adjust_contrast, adjust_brightness

from mmcls.datasets.builder import PIPELINES


@PIPELINES.register_module()
class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.15, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, results):
        results['img'] = adjust_brightness(results['img'])
        results['img'] = adjust_contrast(results['img'])
        return results

    def __repr__(self):
        return repr(self.transform)
