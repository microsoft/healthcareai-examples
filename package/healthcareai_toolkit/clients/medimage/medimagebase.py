# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import base64
import logging
from abc import ABC, abstractmethod

import numpy as np
from healthcareai_toolkit.clients.retry import RetryClient
from healthcareai_toolkit.data.file_types import get_filetype
from healthcareai_toolkit.data.image_preprocessor import ImagePreprocessor
from healthcareai_toolkit.data.image_reader import ImageReader
from healthcareai_toolkit.data.io import numpy_to_image_bytearray
from healthcareai_toolkit.util.azureml_managers import WorkspaceEndpointManager
from healthcareai_toolkit.util.parallel import ParallelSubmitter


def read_file_to_bytes(file):
    with open(file, "rb") as f:
        return f.read()


class MedImageBaseClient(ABC):

    default_engine = "sitk"
    default_normalization = {}
    default_transform = {"pad_to_square": False, "resize_size": None}

    output_dtype = "uint8"
    conversion_mime_types = ["image/png"]

    ENDPOINT_NAME_SETTING = None

    def __init__(
        self,
        endpoint_name=None,
        credential=None,
        params=None,
        reader_params=None,
        normalization_params=None,
        transform_params=None,
        retry_params=None,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        if endpoint_name is None and self.ENDPOINT_NAME_SETTING is not None:
            from healthcareai_toolkit import settings

            endpoint_name = settings.get(self.ENDPOINT_NAME_SETTING)

        self.endpoint_manager = WorkspaceEndpointManager(endpoint_name, credential)
        self.retry_client = RetryClient(retry_params=retry_params, logger=self.logger)
        self.image_reader = self.get_reader(reader_params)
        self.image_preprocessor = self.get_preprocessor(
            normalization_params, transform_params
        )
        self.params = params or {}

        self.logger.debug(
            "%s initialized with endpoint: %s", self.__class__.__name__, endpoint_name
        )

    def get_reader(self, reader_params):
        reader_params = reader_params or {}
        reader_params = {"engine": self.default_engine, **reader_params}
        self.logger.debug("Initializing ImageReader with params: %s", reader_params)
        return ImageReader(**reader_params)

    def get_preprocessor(self, normalization_params, transform_params):
        normalization_params = normalization_params or self.default_normalization
        transform_params = transform_params or self.default_transform
        preprocess_params = {**normalization_params, **transform_params}
        self.logger.debug(
            "Initializing ImagePreprocessor with params: %s", preprocess_params
        )
        return ImagePreprocessor(**preprocess_params)

    @abstractmethod
    def create_payload(self, **kwargs):
        pass

    def decode_response(self, response):
        return response

    def submit(self, **kwargs):
        """
        Submit multiple sets of inputs in a batch.

        Args:
            **kwargs: Lists of inputs, such as image files or text files.

        Returns:
            Batch submission results.
        """
        payload = self.create_payload(**kwargs)
        response = self._submit_payload(payload)
        result = self.decode_response(response)
        return result

    def read_to_image_array(self, image_ref, mime_type=None, reader_overrides=None):
        if isinstance(image_ref, (str, bytearray, bytes)):
            self.logger.debug("Reading image.")
            mime_type = mime_type or get_filetype(image_ref)
            self.logger.debug("Image mime type: %s", mime_type)
            return self.image_reader.read_to_image_array(image_ref, mime_type)
        elif isinstance(image_ref, np.ndarray):
            self.logger.debug("Image determined to be ndarray.")
            return image_ref
        else:
            raise ValueError("Unsupported type!")

    def preprocess_image(
        self, image: np.ndarray, normalization_overrides=None, transform_overrides=None
    ):
        self.logger.debug(
            "Preprocessing image with shape: %s, dtype: %s, min: %s, max: %s",
            image.shape,
            image.dtype,
            image.min(),
            image.max(),
        )
        dtype_info = np.iinfo(self.output_dtype)
        val_min, val_max = dtype_info.min, dtype_info.max

        normalize = str(image.dtype) != self.output_dtype
        image = self.image_preprocessor.preprocess_image(
            image, normalize, normalization_overrides, transform_overrides
        )
        image = np.clip(image, val_min, val_max).astype(self.output_dtype)
        self.logger.debug(
            "Final image with shape: %s, dtype: %s, min: %s, max: %s",
            image.shape,
            image.dtype,
            image.min(),
            image.max(),
        )
        return image

    def read_and_normalize_image(
        self,
        image_ref,
        mime_type=None,
        normalization_overrides=None,
        transform_overrides=None,
    ):
        self.logger.debug("Reading and normalizing image.")
        image = self.read_to_image_array(image_ref, mime_type)
        self.logger.debug(
            "Image read into dtype: %s, min: %s, max: %s",
            image.dtype,
            image.min(),
            image.max(),
        )
        image = self.preprocess_image(
            image, normalization_overrides, transform_overrides
        )
        return image

    def encode_image(self, image):
        self.logger.debug(
            "Encoding image with dtype: %s, min: %s, max: %s",
            image.dtype,
            image.min(),
            image.max(),
        )
        return base64.encodebytes(numpy_to_image_bytearray(image, format="PNG")).decode(
            "utf-8"
        )

    def _read_and_encode_choice(self, input):
        if isinstance(input, dict):
            return self._read_and_encode_image(**input)
        return self._read_and_encode_image(input)

    def _read_and_encode_image(
        self, image, normalization_overrides=None, transform_overrides=None
    ):
        image = self.read_and_normalize_image(
            image,
            normalization_overrides=normalization_overrides,
            transform_overrides=transform_overrides,
        )
        return self.encode_image(image)

    def _read_text_file(self, text_file):
        with open(text_file, "r") as f:
            return f.read()

    def _submit_payload(self, payload):
        self.logger.debug("Submitting payload")
        response = self.retry_client.submit_payload(
            payload,
            target=self.endpoint_manager.target,
            headers=self.endpoint_manager.headers,
        )
        self.logger.debug("Payload Returned")
        return response

    def create_submitter(self, **kwargs):
        """
        Create a BatchSubmitter instance configured with this client's settings.

        Args:
            **kwargs: Keyword arguments for BatchSubmitter.

        Returns:
            BatchSubmitter: Configured BatchSubmitter instance.
        """
        return ParallelSubmitter(self.submit, **kwargs)
