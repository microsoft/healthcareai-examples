# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Pytest conversion of ``test-mii-client-parallel-generator.ipynb``.

Exercises the parallel submitter of :class:`MedImageInsightClient` using the
``generator_unordered`` mode. Results are validated in-memory rather than by
reading a precomputed embeddings artifact, which avoids the stale-schema
``KeyError: 'path'`` seen in the notebook smoke tests.
"""

import glob
import os

import numpy as np
import pandas as pd
import pytest

from healthcareai_toolkit.clients import MedImageInsightClient

pytestmark = pytest.mark.mi2

LIMIT = 5


def _collect_dicom_files(data_root, limit=LIMIT):
    files = [
        f
        for f in glob.glob(os.path.join(data_root, "**", "*.dcm"), recursive=True)
        if os.path.isfile(f)
    ]
    return files[:limit]


def test_parallel_generator_submit(mi2_endpoint, parallel_data_root):
    """The unordered generator yields one embedding per submitted file."""
    files = _collect_dicom_files(parallel_data_root)
    if not files:
        pytest.skip("no .dcm files found under the parallel test data root")

    client = MedImageInsightClient(mi2_endpoint)
    submitter = client.create_submitter(return_as="generator_unordered")

    rows = []
    seen_indices = set()
    for index, result in submitter.submit(image_list=files, total=len(files)):
        assert 0 <= index < len(files)
        assert "image_features" in result
        assert len(result["image_features"]) > 0

        seen_indices.add(index)
        path = os.path.relpath(files[index], parallel_data_root)
        rows.append(
            {
                "path": path,
                "test": path.startswith("test"),
                "inlier": path.startswith(("ref", "test/inlier")),
                **result,
            }
        )

    # Every input must be accounted for exactly once.
    assert seen_indices == set(range(len(files)))
    assert len(rows) == len(files)

    # Build the DataFrame the notebook produced and confirm its schema, which
    # is where the original stale-artifact KeyError surfaced.
    df = pd.DataFrame(rows)
    assert "path" in df.columns
    assert "image_features" in df.columns

    df["image_features"] = df["image_features"].apply(lambda x: np.array(x))
    assert all(features.size > 0 for features in df["image_features"])
