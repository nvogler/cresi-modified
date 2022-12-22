import os
import sys
import skimage.io
import numpy as np

# import imageio

# import relative paths
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataset.abstract_image_type import AbstractImageType


class RawImageType(AbstractImageType):
    """
    image provider constructs image of type and then you can work with it
    """

    def __init__(self, paths, fn, fn_mapping, has_alpha, num_channels):
        super().__init__(paths, fn, fn_mapping, has_alpha)
        if num_channels == 3:
            self.im = skimage.io.imread(os.path.join(self.paths["images"], self.fn))

    def read_image(self, verbose=False):

        im = self.im[..., :-1] if self.has_alpha else self.im

        return self.finalyze(im)

    def read_mask(self, verbose=False):
        path = os.path.join(self.paths["masks"], self.fn_mapping["masks"](self.fn))
        # AVE edit:
        mask_channels = skimage.io.imread(path)
        # skimage reads in (channels, h, w) for multi-channel
        # assume less than 20 channels
        if mask_channels.shape[0] < 20:
            mask = np.moveaxis(mask_channels, 0, -1)
        else:
            mask = mask_channels

        ## original version (mode='L' is a grayscale black and white image)
        return self.finalyze(mask)

    def read_alpha(self):
        return self.finalyze(self.im[..., -1])

    def finalyze(self, data):
        return self.reflect_border(data)
