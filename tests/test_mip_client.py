# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Pytest conversion of ``test-mip-client.ipynb``.

Exercises the :class:`MedImageParseClient` against a live MIP endpoint using a
segmentation example image. The test is skipped automatically when the endpoint
or the example data are not available.
"""

import glob
import itertools
import os

import numpy as np
import pytest

from healthcareai_toolkit.clients import MedImageParseClient

pytestmark = pytest.mark.mip

N = 1
EXTENSIONS = ["jpg", "dcm", "png", "jpeg"]


def _collect_segmentation_files(data_root, n=N):
    """Collect up to ``n`` segmentation-example files per extension."""
    pattern = os.path.join(data_root, "segmentation-examples", "*.*")
    files = [f for f in glob.glob(pattern, recursive=True) if os.path.isfile(f)]
    files_by_ext = {
        k: list(v) for k, v in itertools.groupby(files, lambda x: x.split(".")[-1])
    }
    selected = []
    for ext in EXTENSIONS:
        selected.extend(files_by_ext.get(ext, [])[:n])
    return selected


def _pick_image(files, keyword="covid"):
    """Prefer a file matching ``keyword``, otherwise fall back to the first."""
    matches = [f for f in files if keyword in os.path.basename(f)]
    if matches:
        return matches[0]
    return files[0] if files else None


def test_read_and_normalize_image(mip_endpoint, data_root):
    """Reading and normalizing produces a 2D image array."""
    client = MedImageParseClient(mip_endpoint)
    files = _collect_segmentation_files(data_root)
    image_file = _pick_image(files)
    if image_file is None:
        pytest.skip("no segmentation-example images found under DATA_ROOT")

    image = client.read_and_normalize_image(image_file)
    assert isinstance(image, np.ndarray)
    assert image.ndim >= 2


def test_submit_returns_masks(mip_endpoint, data_root):
    """Submitting an image with a prompt returns a mask matching the image."""
    client = MedImageParseClient(mip_endpoint)
    files = _collect_segmentation_files(data_root)
    image_file = _pick_image(files)
    if image_file is None:
        pytest.skip("no segmentation-example images found under DATA_ROOT")

    image = client.read_and_normalize_image(image_file)
    result = client.submit(image_list=[image_file], prompts=["lung"])

    assert len(result) == 1
    masks = result[0]["image_features"]
    assert isinstance(masks, np.ndarray)
    assert len(masks) >= 1
    # Each mask spatial dimension should match the input image.
    assert masks[0].shape[:2] == image.shape[:2]
