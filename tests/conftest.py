import os

import pytest

_MARKERS = {
    "mi2": "notebooks that call the MedImageInsight (MI2) endpoint",
    "mip": "notebooks that call the MedImageParse (MIP) endpoint",
    "pgp": "notebooks that call the Prov-GigaPath endpoint",
    "gpt": "notebooks that call Azure OpenAI",
    "gigatime": "notebooks that call the GigaTime endpoint",
}


def pytest_configure(config):
    for name, description in _MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {description}")


def _require_env(var):
    """Skip the test if the given environment variable is not set."""
    value = os.environ.get(var)
    if not value:
        pytest.skip(f"{var} not set")
    return value


@pytest.fixture
def mi2_endpoint():
    """MedImageInsight endpoint, skipping the test when it is not configured."""
    _require_env("MI2_MODEL_ENDPOINT")
    from healthcareai_toolkit import settings

    return settings.MI2_MODEL_ENDPOINT


@pytest.fixture
def mip_endpoint():
    """MedImageParse endpoint, skipping the test when it is not configured."""
    _require_env("MIP_MODEL_ENDPOINT")
    from healthcareai_toolkit import settings

    return settings.MIP_MODEL_ENDPOINT


@pytest.fixture
def data_root():
    """Example data root, skipping the test when it is not available."""
    from healthcareai_toolkit import settings

    root = settings.DATA_ROOT
    if not root or not os.path.isdir(root):
        pytest.skip("DATA_ROOT is not available")
    return root


@pytest.fixture
def parallel_data_root():
    """Data root used by the parallel generator test.

    Prefers ``PARALLEL_TEST_DATA_ROOT`` and falls back to ``DATA_ROOT`` so the
    test can still exercise the parallel submitter when only the example data
    repository is available.
    """
    from healthcareai_toolkit import settings

    root = settings.PARALLEL_TEST_DATA_ROOT or settings.DATA_ROOT
    if not root or not os.path.isdir(root):
        pytest.skip("no parallel test data root available")
    return root
