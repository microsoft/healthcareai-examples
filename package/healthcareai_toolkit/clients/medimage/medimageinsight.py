# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from itertools import zip_longest
from .medimagebase import MedImageBaseClient


class MedImageInsightClient(MedImageBaseClient):

    default_normalization = {"percentiles": (0.01, 0.99)}
    default_transform = {
        "pad_to_square": False,
        "resize_size": (480, 480),
        "resize_edge": False,
    }
    ENDPOINT_NAME_SETTING = "MI2_MODEL_ENDPOINT"

    def create_payload(self, image_list=None, text_list=None, text_file_list=None):

        image_list = image_list or []
        text_list = text_list or []
        text_file_list = text_file_list or []

        if len(text_file_list) and len(text_list):
            raise ValueError(
                "Only one of text_file_list or text_list should be provided"
            )

        if len(text_file_list):
            text_list = [self._read_text_file(tf) for tf in text_file_list]
        image_list = [self._read_and_encode_choice(image) for image in image_list]
        iter = list(zip_longest(image_list, text_list))
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
