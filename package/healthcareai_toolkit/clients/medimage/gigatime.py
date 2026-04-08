# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from itertools import zip_longest

import numpy as np

from .medimagebase import MedImageBaseClient


class GigaTimeClient(MedImageBaseClient):
    ENDPOINT_NAME_SETTING = "GIGATIME_MODEL_ENDPOINT"

    def create_payload(self, image_list=None):
        image_list = image_list or []

        image_list = [self._read_and_encode_choice(image) for image in image_list]
        payload_rows = list(zip_longest(image_list))

        payload = {
            "input_data": {
                "columns": ["image"],
                "index": [i for i in range(len(payload_rows))],
                "data": [list(data) for data in payload_rows],
            }
        }
        return payload

    def decode_response(self, response):
        response = super().decode_response(response)

        if isinstance(response, dict) and "predictions" in response:
            response = response["predictions"]

        predictions = np.asarray(response, dtype=np.float32)
        if predictions.ndim == 3:
            predictions = predictions[np.newaxis, ...]

        if predictions.ndim != 4:
            raise ValueError(
                f"Unexpected GigaTime endpoint output shape: {predictions.shape}"
            )

        return [prediction for prediction in predictions]
