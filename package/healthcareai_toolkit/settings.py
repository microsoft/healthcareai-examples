import os
import types
import re

from dotenv import load_dotenv

load_dotenv()

MI2_MODEL_ENDPOINT = os.environ.get("MI2_MODEL_ENDPOINT", None)
MIP_MODEL_ENDPOINT = os.environ.get("MIP_MODEL_ENDPOINT", None)
GIGAPATH_MODEL_ENDPOINT = os.environ.get("GIGAPATH_MODEL_ENDPOINT", None)
CXRREPORTGEN_MODEL_ENDPOINT = os.environ.get("CXRREPORTGEN_MODEL_ENDPOINT", None)

DATA_ROOT = os.environ.get("DATA_ROOT", "/home/azureuser/data/healthcare-ai/")

PARALLEL_TEST_DATA_ROOT = os.environ.get("PARALLEL_TEST_DATA_ROOT", None)


def _get_azure_openai_config():
    """
    Get Azure OpenAI configuration from environment variables.
    """
    endpoint_url = os.environ.get("AZURE_OPENAI_ENDPOINT", None)

    if not endpoint_url:
        return None, None, None

    # Validate that endpoint_url is a valid URL
    if not endpoint_url.startswith(("http://", "https://")):
        raise ValueError(
            f"AZURE_OPENAI_ENDPOINT must be a valid URL starting with http:// or https://, "
            f"got: {endpoint_url}"
        )

    # Try to parse as inference URI
    # Format: https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version={version}
    match = re.search(
        r"(?P<endpoint>https://[^/]+)/openai/deployments/(?P<deployment>[^/]+)/.*api-version=(?P<api_version>[^&]+)",
        endpoint_url,
    )
    if match:
        return (
            match.group("endpoint"),
            match.group("deployment"),
            match.group("api_version"),
        )

    # Base endpoint format - use separate environment variables
    deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", None)
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    # Raise error if base endpoint is set but missing deployment name
    if not deployment_name:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT is set to a base endpoint, but AZURE_OPENAI_DEPLOYMENT_NAME "
            "is required. Either provide both values or use a full inference URI format."
        )

    return endpoint_url, deployment_name, api_version


# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", None)
(
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_OPENAI_API_VERSION,
) = _get_azure_openai_config()


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
