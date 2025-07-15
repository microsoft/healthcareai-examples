# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .medimagebase import MedImageBaseClient
import json


class CxrReportGenClient(MedImageBaseClient):
    """
    Client for generating chest X-ray report payloads.
    """

    default_normalization = {"percentiles": (0.01, 0.99)}
    ENDPOINT_NAME_SETTING = "CXRREPORTGEN_MODEL_ENDPOINT"

    def create_payload(
        self,
        frontal_image: str,
        indication: str,
        technique: str,
        comparison: str = None,
        lateral_image: str = None,
        prior_image: str = None,
        prior_report: str = None,
    ) -> dict:
        input_data = {
            "frontal_image": self._read_and_encode_choice(frontal_image),
            "indication": indication,
            "technique": technique,
            "comparison": comparison if comparison else "None",
        }
        if lateral_image:
            input_data["lateral_image"] = self._read_and_encode_choice(lateral_image)
        if prior_image:
            input_data["prior_image"] = self._read_and_encode_choice(prior_image)
        if prior_report:
            input_data["prior_report"] = prior_report

        payload = {
            "input_data": {
                "columns": list(input_data.keys()),
                "index": [0],
                "data": [
                    list(input_data.values()),
                ],
            },
            "params": {},
        }
        return payload

    def decode_response(self, response):
        response = super().decode_response(response)
        return [{**r, "output": json.loads(r["output"])} for r in response]
