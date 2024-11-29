# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import base64
import io
import os
import tempfile
import warnings
from io import BytesIO
import glob
import re

from typing import Union

import nibabel as nib
import numpy as np
import pydicom
import requests
import SimpleITK as sitk
from PIL import Image

from tqdm import tqdm
from collections.abc import Iterable

import magic


def read_dicom_to_array(dicom_input, engine="sitk"):
    """Reads a DICOM file or bytes and returns the image array using the specified engine."""
    if engine not in ["pydicom", "sitk"]:
        raise ValueError("Unsupported engine. Use 'pydicom' or 'sitk'.")

    if isinstance(dicom_input, bytes) and engine == "sitk":
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as temp_file:
            temp_file.write(dicom_input)
            temp_file.flush()
            dicom_input = temp_file.name

    if engine == "sitk":
        return _read_dicom_sitk(dicom_input)
    elif engine == "pydicom":
        return _read_dicom_pydicom(dicom_input)


def _read_dicom_sitk(dicom_input, squeeze=True, suppress_warnings=False):
    try:
        if suppress_warnings:
            original_warning_state = sitk.ProcessObject_GetGlobalWarningDisplay()
            sitk.ProcessObject_SetGlobalWarningDisplay(False)
        image = sitk.ReadImage(dicom_input)
        image_array = sitk.GetArrayFromImage(image).astype(np.float32)

        if squeeze:
            if not suppress_warnings and image_array.shape[0] > 1:
                warnings.warn(
                    f"Squeezing the first dimension of size {image_array.shape[0]}"
                )
            image_array = image_array[0, :, :]

        return image_array
    finally:
        if suppress_warnings:
            sitk.ProcessObject_SetGlobalWarningDisplay(original_warning_state)


def _read_dicom_pydicom(dicom_input):
    ds = pydicom.dcmread(dicom_input)
    rescale_slope = getattr(ds, "RescaleSlope", 1)
    rescale_intercept = getattr(ds, "RescaleIntercept", 0)
    image_array = ds.pixel_array * rescale_slope + rescale_intercept
    return image_array


def normalize_image_to_uint8(image_array, window=None, percentiles=None, min_max=None):
    """Normalize a DICOM image array to uint8 format.

    Args:
        image_array (np.ndarray): The input image array.
        window (tuple, optional): A tuple (window_center, window_width) for windowing.
        percentiles (tuple or float, optional): A tuple (low_percentile, high_percentile) or a single float for percentile normalization.
        min_max (tuple, optional): A tuple (min_val, max_val) for min-max normalization. If None, use the image array's min and max.

    Returns:
        np.ndarray: The normalized image array in uint8 format.
    """
    # Ensure only one of the optional parameters is provided
    if sum([window is not None, percentiles is not None, min_max is not None]) > 1:
        raise ValueError(
            "Only one of 'window', 'percentiles', or 'min_max' must be specified."
        )

    if window:
        window_center, window_width = window
        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2
    elif percentiles:
        if isinstance(percentiles, float):
            low_percentile, high_percentile = percentiles, 1 - percentiles
        else:
            low_percentile, high_percentile = percentiles
        min_val = np.percentile(image_array, low_percentile * 100)
        max_val = np.percentile(image_array, high_percentile * 100)
    elif min_max is not None and isinstance(min_max, Iterable) and len(min_max) == 2:
        min_val, max_val = min_max
    else:
        min_val = np.min(image_array)
        max_val = np.max(image_array)

    image_array = np.clip(image_array, min_val, max_val)
    image_array = (image_array - min_val) / (max_val - min_val) * 255.0

    return image_array.astype(np.uint8)


def numpy_to_image_bytearray(image_array: np.ndarray, format: str = "PNG") -> bytes:
    """Convert a NumPy array to an image byte array."""
    byte_io = BytesIO()
    pil_image = Image.fromarray(image_array)
    if pil_image.mode == "L":
        pil_image = pil_image.convert("RGB")
    pil_image.save(byte_io, format=format)
    return byte_io.getvalue()


def read_rgb_to_array(image_path):
    """Reads an RGB image and returns resized pixel data as a BytesIO buffer."""
    image_array = np.array(Image.open(image_path))
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    return image_array


def read_image_to_array(input_data: Union[bytes, str]) -> np.ndarray:
    """Reads an image from a file path or bytes and returns it as a numpy array."""
    if isinstance(input_data, bytes):
        return np.array(Image.open(BytesIO(input_data)))
    elif isinstance(input_data, str):
        return np.array(Image.open(input_data))
    else:
        raise ValueError("Input must be a file path (str) or file bytes.")


def read_file_to_bytes(input_data: Union[bytes, str]) -> bytes:
    """Reads a file from a file path or returns file bytes directly."""
    if isinstance(input_data, bytes):
        return input_data
    elif isinstance(input_data, str):
        with open(input_data, "rb") as f:
            return f.read()
    else:
        raise ValueError("Input must be a file path (str) or file bytes.")


def read_nifti(image_path, slice_idx=0, HW_index=(0, 1), channel_idx=None):
    """Reads a NIFTI file and returns pixel data as a BytesIO buffer."""
    nii = nib.load(image_path)
    image_array = nii.get_fdata()

    if HW_index != (0, 1):
        image_array = np.moveaxis(image_array, HW_index, (0, 1))

    if channel_idx is None:
        image_array = image_array[:, :, slice_idx]
    else:
        image_array = image_array[:, :, slice_idx, channel_idx]

    return image_array
