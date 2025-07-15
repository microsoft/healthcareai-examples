import os
import types

from dotenv import load_dotenv

load_dotenv()

MI2_MODEL_ENDPOINT = os.environ.get("MI2_MODEL_ENDPOINT", None)
MIP_MODEL_ENDPOINT = os.environ.get("MIP_MODEL_ENDPOINT", None)
GIGAPATH_MODEL_ENDPOINT = os.environ.get("GIGAPATH_MODEL_ENDPOINT", None)
CXRREPORTGEN_MODEL_ENDPOINT = os.environ.get("CXRREPORTGEN_MODEL_ENDPOINT", None)

DATA_ROOT = os.environ.get("DATA_ROOT", "/home/azureuser/data/healthcare-ai/")

PARALLEL_TEST_DATA_ROOT = os.environ.get("PARALLEL_TEST_DATA_ROOT", None)

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", None)
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", None)
AZURE_OPENAI_MODEL_NAME = os.environ.get("AZURE_OPENAI_MODEL_NAME", None)


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
