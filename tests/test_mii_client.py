# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Pytest conversion of ``test-mii-client.ipynb``.

Exercises the :class:`MedImageInsightClient` against a live MI2 endpoint using
a small selection of example images. The test is skipped automatically when the
endpoint or the example data are not available.
"""

import glob
import itertools
import os

import numpy as np
import pytest

from healthcareai_toolkit.clients import MedImageInsightClient

pytestmark = pytest.mark.mi2

# Number of files to pick per extension, mirroring the original notebook.
N = 1
EXTENSIONS = ["jpg", "dcm", "png", "jpeg"]


def _collect_example_files(data_root, n=N):
    """Collect up to ``n`` files per extension from ``data_root``."""
    files = [
        f
        for f in glob.glob(os.path.join(data_root, "**", "*.*"), recursive=True)
        if os.path.isfile(f)
    ]
    files_by_ext = {
        k: list(v) for k, v in itertools.groupby(files, lambda x: x.split(".")[-1])
    }
    selected = []
    for ext in EXTENSIONS:
        selected.extend(files_by_ext.get(ext, [])[:n])
    return selected


def test_read_to_image_array(mi2_endpoint, data_root):
    """The client can read supported files into numpy image arrays."""
    client = MedImageInsightClient(mi2_endpoint)
    files = _collect_example_files(data_root)
    if not files:
        pytest.skip("no example images found under DATA_ROOT")

    for file in files:
        image = client.read_to_image_array(file)
        assert isinstance(image, np.ndarray)
        assert image.size > 0


def test_submit_returns_image_features(mi2_endpoint, data_root):
    """Submitting images returns an embedding vector for each input."""
    client = MedImageInsightClient(mi2_endpoint)
    images = _collect_example_files(data_root)
    if not images:
        pytest.skip("no example images found under DATA_ROOT")

    result = client.submit(image_list=images)

    assert len(result) == len(images)
    for entry in result:
        assert "image_features" in entry
        features = np.asarray(entry["image_features"])
        assert features.size > 0
        assert np.issubdtype(features.dtype, np.floating)
