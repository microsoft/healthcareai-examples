# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import SimpleITK as sitk
from skimage import measure, transform
from collections.abc import Iterable
from typing import Union, Iterable
from skimage import transform


def extract_instances_from_mask(mask):
    # get intances from binary mask
    seg = sitk.GetImageFromArray(mask)
    filled = sitk.BinaryFillhole(seg)
    d = sitk.SignedMaurerDistanceMap(
        filled, insideIsPositive=False, squaredDistance=False, useImageSpacing=False
    )

    ws = sitk.MorphologicalWatershed(d, markWatershedLine=False, level=1)
    ws = sitk.Mask(ws, sitk.Cast(seg, ws.GetPixelID()))
    ins_mask = sitk.GetArrayFromImage(ws)

    # filter out instances with small area outliers
    props = measure.regionprops_table(ins_mask, properties=("label", "area"))
    mean_area = np.mean(props["area"])
    std_area = np.std(props["area"])

    threshold = mean_area - 2 * std_area - 1
    ins_mask_filtered = ins_mask.copy()
    for i, area in zip(props["label"], props["area"]):
        if area < threshold:
            ins_mask_filtered[ins_mask == i] = 0

    return ins_mask_filtered


def pad_to_square(image):
    """Pads the image to make it square with equal padding on both sides."""
    shape = image.shape
    if shape[0] > shape[1]:
        pad = (shape[0] - shape[1]) // 2
        pad_width = ((0, 0), (pad, pad))
    elif shape[0] < shape[1]:
        pad = (shape[1] - shape[0]) // 2
        pad_width = ((pad, pad), (0, 0))
    else:
        pad_width = None

    if pad_width is not None:
        if image.ndim == 3:
            pad_width += ((0, 0),)
        image = np.pad(image, pad_width, "constant", constant_values=0)
    return image


def resize_image(
    image: np.ndarray, size: Union[int, Iterable[int]] = 1024
) -> np.ndarray:
    """Resizes the image to the given size. Handles both 2D and 3D images.

    If size is an iterable (like a tuple or list), the image is resized to that shape (height, width).
    If it is an int, the image is resized to (size, size).
    """
    if isinstance(size, Iterable) and not isinstance(size, (str, bytes)):
        target_shape: tuple = tuple(size)
    elif isinstance(size, int):
        target_shape = (size, size)
    else:
        raise ValueError("size must be an int or an iterable (tuple/list) of two ints")

    def resize_slice(slice: np.ndarray) -> np.ndarray:
        return transform.resize(
            slice,
            target_shape,
            order=3,
            mode="constant",
            preserve_range=True,
            anti_aliasing=True,
        )

    if image.ndim == 2:
        return np.stack([resize_slice(image)] * 3, axis=2)
    elif image.ndim == 3:
        return np.stack(
            [resize_slice(image[:, :, i]) for i in range(image.shape[2])], axis=2
        )
    else:
        raise ValueError("Unsupported image dimensions: {}".format(image.ndim))


def extract_instance_masks(mask, threshold=None):
    """
    Extracts instance masks from a binary mask using morphological watershed segmentation.

    Args:
        mask (numpy.ndarray): A binary mask array where the instances are to be extracted.
        threshold (float, optional): Threshold value to filter out small area instances.
            If None, threshold is calculated as (mean_area - 2 * std_area - 1).

    Returns:
        numpy.ndarray: An array with the extracted instance masks, where small area outliers are filtered out.
    """
    # get instances from binary mask
    seg = sitk.GetImageFromArray(mask)
    filled = sitk.BinaryFillhole(seg)
    d = sitk.SignedMaurerDistanceMap(
        filled, insideIsPositive=False, squaredDistance=False, useImageSpacing=False
    )

    ws = sitk.MorphologicalWatershed(d, markWatershedLine=False, level=1)
    ws = sitk.Mask(ws, sitk.Cast(seg, ws.GetPixelID()))
    ins_mask = sitk.GetArrayFromImage(ws)

    # filter out instances with small area outliers
    props = measure.regionprops_table(ins_mask, properties=("label", "area"))
    areas = np.array(props["area"])

    if threshold is None:
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        threshold = mean_area - 2 * std_area - 1

    ins_mask_filtered = ins_mask.copy()
    for i, area in zip(props["label"], areas):
        if area < threshold:
            ins_mask_filtered[ins_mask == i] = 0

    return ins_mask_filtered
