# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import base64
import json
from itertools import zip_longest

import numpy as np

from .medimagebase import MedImageBaseClient


def _decode_image_features(json_encoded):
    """Decode an image pixel data array in JSON.
    Return image pixel data as an array.
    """
    array_metadata = json.loads(json_encoded)
    base64_encoded = array_metadata["data"]
    shape = tuple(array_metadata["shape"])
    dtype = np.dtype(array_metadata["dtype"])
    array_bytes = base64.b64decode(base64_encoded)
    array = np.frombuffer(array_bytes, dtype=dtype).reshape(shape)
    return array


class MedImageParseClient(MedImageBaseClient):

    special_normalizations = {
        "ct-abdomen": {"min_max": [-150, 250]},
        "ct-lung": {"min_max": [-1000, 1000]},
        "ct-pelvis": {"min_max": [-55, 200]},
        "ct-liver": {"min_max": [-25, 230]},
        "ct-colon": {"min_max": [-68, 187]},
        "ct-pancreas": {"min_max": [-100, 200]},
    }

    default_normalization = {"percentiles": (0.005, 0.995)}
    default_transform = {"pad_to_square": True, "resize_size": 1024}

    ENDPOINT_NAME_SETTING = "MIP_MODEL_ENDPOINT"

    def __init__(
        self,
        endpoint_name=None,
        credential=None,
        engine="sitk",
        special_normalization=None,
        params=None,
        normalization_params=None,
        retry_params=None,
    ):

        if special_normalization is not None:
            normalization_params = self.get_special_normalization(special_normalization)
        super().__init__(
            endpoint_name,
            credential,
            engine,
            params,
            normalization_params,
            retry_params,
        )

    def get_special_normalization(self, special_normalization):
        if special_normalization.lower() not in self.special_normalizations:
            raise ValueError(f"unknown task type {special_normalization}")
        return self.special_normalizations[special_normalization]

    def create_payload(self, image_list=None, prompts=None):
        image_list = image_list or []
        prompts = prompts or []
        image_list = [self._read_and_encode_choice(image) for image in image_list]
        iter = list(zip_longest(image_list, prompts))
        max_len = len(iter)

        payload = {
            "input_data": {
                "columns": ["image", "text"],
                "index": [i for i in range(max_len)],
                "data": [list(data) for data in iter],
            },
            "params": self.params,
        }
        return payload

    def read_and_normalize_image(
        self,
        image_ref,
        mime_type=None,
        normalization_overrides=None,
        transform_overrides=None,
        special_overrides=None,
    ):
        if normalization_overrides is not None and special_overrides is not None:
            raise ValueError("Cannot pass both normalization_params and special params")
        if special_overrides is not None:
            normalization_overrides = self.get_special_normalization(special_overrides)
        return super().read_and_normalize_image(
            image_ref,
            mime_type=mime_type,
            normalization_overrides=normalization_overrides,
            transform_overrides=transform_overrides,
        )

    def decode_response(self, response):
        response = super().decode_response(response)
        return [
            {**r, "image_features": _decode_image_features(r["image_features"])}
            for r in response
        ]
