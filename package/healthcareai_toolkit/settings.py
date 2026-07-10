import os
import re
import types

from dotenv import load_dotenv

load_dotenv()

_INFERENCE_URI_RE = re.compile(
    r"(?P<endpoint>https://[^/]+)/openai/deployments/(?P<deployment>[^/]+)/.*api-version=(?P<api_version>[^&]+)"
)


def _get_azure_openai_config():
    """Parse Azure OpenAI config from environment. Returns (endpoint, deployment, api_version) tuple.
    Returns (None, None, None) if AZURE_OPENAI_ENDPOINT is not set."""
    endpoint_url = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not endpoint_url:
        return (None, None, None)
    match = _INFERENCE_URI_RE.search(endpoint_url)
    if match:
        return (
            match.group("endpoint"),
            match.group("deployment"),
            match.group("api_version"),
        )
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    return (endpoint_url, deployment, api_version)


MI2_MODEL_ENDPOINT = os.environ.get("MI2_MODEL_ENDPOINT", None)
MIP_MODEL_ENDPOINT = os.environ.get("MIP_MODEL_ENDPOINT", None)
GIGAPATH_MODEL_ENDPOINT = os.environ.get("GIGAPATH_MODEL_ENDPOINT", None)
GIGATIME_MODEL_ENDPOINT = os.environ.get("GIGATIME_MODEL_ENDPOINT", None)
CXRREPORTGEN_MODEL_ENDPOINT = os.environ.get("CXRREPORTGEN_MODEL_ENDPOINT", None)

DATA_ROOT = os.environ.get("DATA_ROOT", "/home/azureuser/data/healthcare-ai/")

PARALLEL_TEST_DATA_ROOT = os.environ.get("PARALLEL_TEST_DATA_ROOT", None)

DISABLE_MANAGED_IDENTITY = os.environ.get(
    "AZURE_IDENTITY_DISABLE_MANAGED_IDENTITY", ""
).lower() in ("true", "1")

AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", None)
(AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_VERSION) = (
    _get_azure_openai_config()
)


_constants = {
    name: value
    for name, value in globals().items()
    if not name.startswith("_")
    and not isinstance(value, types.FunctionType)
    and not isinstance(value, types.ModuleType)
    and not isinstance(value, type)
}


def keys():
    return _constants.keys()


def get(name):
    if name in keys():
        return _constants[name]
    else:
        raise KeyError(f"'{name}' is not a valid constant in settings.")
