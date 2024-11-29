# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from typing import Union
import re

from .file_types import get_filetype
from .io import read_dicom_to_array, normalize_image_to_uint8, read_image_to_array


class ImageReader:
    known_file_types = [r"application/dicom", r"image/.*"]

    def __init__(self, engine="sitk"):
        self.engine = engine

    def _validate_mime_type(self, mime_type):
        if not any(re.match(pattern, mime_type) for pattern in self.known_file_types):
            raise ValueError(f"Unsupported file type: {mime_type}")

    def read_to_image_array(self, input_data, mime_type=None, reader_overrides=None):
        mime_type = mime_type or get_filetype(input_data)

        self._validate_mime_type(mime_type)
        if mime_type == "application/dicom":
            image_array = read_dicom_to_array(input_data, engine=self.engine)
        elif mime_type == "application/nifti":
            raise NotImplementedError(f"nifti not supported yet")
        elif mime_type.startswith("image"):
            image_array = read_image_to_array(input_data)
        else:
            raise ValueError("Unknown or unsupported file type.")
        return image_array
