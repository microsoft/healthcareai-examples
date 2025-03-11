# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import numpy as np

from healthcareai_toolkit.data.manip import (
    resize_image,
    pad_to_square as _pad_to_square,
)
from healthcareai_toolkit.data.io import normalize_image_to_uint8


class ImagePreprocessor:
    def __init__(
        self,
        pad_to_square=False,
        resize_size=None,
        resize_edge=False,
        window=None,
        percentiles=None,
        min_max=None,
    ):
        self.pad_to_square = pad_to_square
        self.resize_size = resize_size
        self.resize_edge = resize_edge
        self.window = window
        self.percentiles = percentiles
        self.min_max = min_max
        self.logger = logging.getLogger(self.__class__.__name__)

    def normalize_image(self, image, normalization_overrides=None):
        params = normalization_overrides or {
            "window": self.window,
            "percentiles": self.percentiles,
            "min_max": self.min_max,
        }
        self.logger.debug(
            f"Normalization overrides passed: {normalization_overrides}, Final: parameters: {params}"
        )
        image = normalize_image_to_uint8(image, **params)
        return image

    def transform_image(self, image, transform_overrides=None):
        params = transform_overrides or {
            "pad_to_square": self.pad_to_square,
            "resize_size": self.resize_size,
            "resize_edge": self.resize_edge,
        }
        self.logger.debug(
            f"Transform overrides passed: {transform_overrides}, Final: parameters: {params}"
        )
        image = self._transform_image(image, **params)
        return image

    @staticmethod
    def _transform_image(
        image, pad_to_square=False, resize_size=None, resize_edge=False
    ):
        if pad_to_square:
            image = _pad_to_square(image)
        if resize_size is not None:
            if resize_edge:
                # Resize so that the largest edge equals resize_size while keeping the aspect ratio.
                h, w = image.shape[:2]
                scale = resize_size / max(h, w)
                new_h = int(h * scale)
                new_w = int(w * scale)
                image = resize_image(image, (new_h, new_w))
            else:
                image = resize_image(image, resize_size)
        return image

    def preprocess_image(
        self, image, normalize=True, normalization_params=None, transform_params=None
    ):
        if normalize:
            image = self.normalize_image(image, normalization_params)
        return self.transform_image(image, transform_params)
