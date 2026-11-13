# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from healthcareai_toolkit import settings

from .medimagebase import MedImageBaseClient


class MedImageInsightPremiumClient(MedImageBaseClient):
    """Embed one image, text, or text file per request.

    Accepts one nonempty singleton list or tuple via image_list, text_list, or
    text_file_list. Shared image preprocessing produces PNG data URIs. params
    pass through for service validation; returned payloads are trusted.
    Returns a list containing embeddings and response metadata; float embeddings
    also map to text_features (when a text echo is present) or image_features.
    Response metadata is preserved; missing scaling_factor values are not filled in.
    """

    ENDPOINT_NAME_SETTING = "MI2_PREMIUM_ENDPOINT"
    API_KEY_SETTING = "MI2_PREMIUM_API_KEY"
    DEPLOYMENT_NAME_SETTING = "MI2_PREMIUM_DEPLOYMENT_NAME"
    DEFAULT_DEPLOYMENT_NAME = "MedImageInsight-Premium"
    API_PATH = "/providers/microsoft/v2/embed"
    default_normalization = {"percentiles": (0.01, 0.99)}
    default_transform = {
        "pad_to_square": False,
        "resize_size": (480, 480),
        "resize_edge": False,
    }

    def __init__(self, *args, deployment_name=None, **kwargs):
        if deployment_name is None:
            deployment_name = (
                settings.get(self.DEPLOYMENT_NAME_SETTING)
                or self.DEFAULT_DEPLOYMENT_NAME
            )
        if deployment_name is None or deployment_name == "":
            raise ValueError(
                f"Set {self.DEPLOYMENT_NAME_SETTING} or pass deployment_name."
            )
        self.deployment_name = deployment_name
        super().__init__(*args, **kwargs)

    def _validate_kwargs(self, image_list=None, text_list=None, text_file_list=None):
        inputs = [value for value in (image_list, text_list, text_file_list) if value]
        if len(inputs) != 1:
            raise ValueError(
                "Provide exactly one nonempty image_list, text_list, or text_file_list."
            )
        if not isinstance(inputs[0], (list, tuple)) or len(inputs[0]) != 1:
            raise ValueError(
                "MI2 Premium accepts one input in a list or tuple; use batch_size=1."
            )

    def create_payload(self, image_list=None, text_list=None, text_file_list=None):
        payload = {"model": self.deployment_name, **self.params}
        if image_list:
            encoded = "".join(self._read_and_encode_choice(image_list[0]).split())
            payload["images"] = [f"data:image/png;base64,{encoded}"]
        else:
            texts = (
                [self._read_text_file(path) for path in text_file_list]
                if text_file_list
                else list(text_list)
            )
            payload["texts"] = texts
        return payload

    def decode_response(self, response):
        embeddings = response["embeddings"]
        count = len(next(iter(embeddings.values())))
        texts = response.get("texts")
        metadata = {
            key: value
            for key, value in response.items()
            if key not in {"embeddings", "texts"}
        }
        result = []
        for index in range(count):
            row = {
                **metadata,
                "embeddings": {
                    kind: values[index] for kind, values in embeddings.items()
                },
            }
            if texts is not None:
                row["text"] = [texts[index]]
            if "float" in embeddings:
                row["text_features" if texts is not None else "image_features"] = [
                    embeddings["float"][index]
                ]
            result.append(row)
        return result
