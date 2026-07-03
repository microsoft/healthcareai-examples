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
