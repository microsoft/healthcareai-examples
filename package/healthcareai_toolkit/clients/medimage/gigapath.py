# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import base64
import json
import io
import torch
from itertools import zip_longest

import numpy as np

from .medimagebase import MedImageBaseClient


def _decode_image_features(feature):
    """Decode a feature map data array in JSON.
    Return feature map data as an array.
    """
    feature_bytes = base64.b64decode(feature)
    buffer = io.BytesIO(feature_bytes)
    tmp = torch.load(buffer, weights_only=True, map_location="cpu")
    feature_output = tmp.cpu().data.numpy()
    return feature_output


class GigaPathClient(MedImageBaseClient):

    default_normalization = {"percentiles": (0.01, 0.99)}
    ENDPOINT_NAME_SETTING = "GIGAPATH_MODEL_ENDPOINT"

    def create_payload(self, image_list=None):
        image_list = image_list or []

        image_list = [self._read_and_encode_choice(image) for image in image_list]
        iter = list(zip_longest(image_list))
        max_len = len(iter)

        payload = {
            "input_data": {
                "columns": ["image"],
                "index": [i for i in range(max_len)],
                "data": [list(data) for data in iter],
            }
        }
        return payload

    def decode_response(self, response):
        response = super().decode_response(response)
        return [
            {**r, "image_features": _decode_image_features(r["image_features"])}
            for r in response
        ]
