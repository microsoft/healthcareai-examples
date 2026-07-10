import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

papermill = pytest.importorskip("papermill")

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")
KERNEL = os.environ.get("PAPERMILL_KERNEL", "healthcareai")

DEFAULT_TIMEOUT = 300

DATA_ROOT = os.environ.get("DATA_ROOT", "/home/azureuser/data/healthcare-ai/")

MANIFEST = {
    "zero-shot-classification": {
        "path": "azureml/medimageinsight/zero-shot-classification.ipynb",
        "endpoints": ["MI2_MODEL_ENDPOINT"],
        "markers": ["mi2"],
        "timeout": 1800,
    },
    "advanced-call-example": {
        "path": "azureml/medimageinsight/advanced-call-example.ipynb",
        "endpoints": ["MI2_MODEL_ENDPOINT"],
        "markers": ["mi2"],
        "timeout": 1800,
    },
    "outlier-detection-demo": {
        "path": "azureml/medimageinsight/outlier-detection-demo.ipynb",
        "endpoints": ["MI2_MODEL_ENDPOINT"],
        "markers": ["mi2"],
    },
    "model-comparison": {
        "path": "azureml/medimageinsight/model-comparison.ipynb",
        "endpoints": ["MI2_MODEL_ENDPOINT"],
        "markers": ["mi2"],
        "timeout": 1800,
    },
    "exam-parameter-detection": {
        "path": "azureml/medimageinsight/exam-parameter-demo/exam-parameter-detection.ipynb",
        "endpoints": ["AZURE_OPENAI_ENDPOINT"],
        "markers": ["gpt"],
    },
    "agent-classification-example": {
        "path": "azureml/medimageinsight/agent-classification-example.ipynb",
        "endpoints": [
            "MI2_MODEL_ENDPOINT",
            "AZURE_OPENAI_ENDPOINT",
        ],
        "markers": ["mi2", "gpt"],
    },
    "adapter-training": {
        "path": "azureml/medimageinsight/adapter-training.ipynb",
        "endpoints": ["MI2_MODEL_ENDPOINT"],
        "markers": ["mi2"],
        "timeout": 1800,
    },
    "medimageparse_segmentation_demo": {
        "path": "azureml/medimageparse/medimageparse_segmentation_demo.ipynb",
        "endpoints": ["MIP_MODEL_ENDPOINT"],
        "markers": ["mip"],
    },
    "virtual_phenotyping": {
        "path": "azureml/gigatime/virtual_phenotyping.ipynb",
        "endpoints": ["GIGATIME_MODEL_ENDPOINT"],
        "markers": ["gigatime"],
    },
    "rad_path_survival_demo": {
        "path": "azureml/advanced_demos/radpath/rad_path_survival_demo.ipynb",
        "endpoints": ["MI2_MODEL_ENDPOINT", "GIGAPATH_MODEL_ENDPOINT"],
        "markers": ["mi2", "pgp"],
    },
    "rag_infection_detection": {
        "path": "azureml/advanced_demos/image_search/rag_infection_detection.ipynb",
        "endpoints": ["MI2_MODEL_ENDPOINT"],
        "markers": ["mi2"],
        # The WoundcareVQA dataset is not redistributed (licensing) and there is
        # no headless download for it yet, so skip unless the data is present.
        # Fixing this notebook is deferred to a follow-up PR.
        "requires_data": ["woundcarevqa"],
    },
    "2d_pathology_image_search": {
        "path": "azureml/advanced_demos/image_search/2d_pathology_image_search.ipynb",
        "endpoints": ["GIGAPATH_MODEL_ENDPOINT"],
        "markers": ["pgp"],
    },
    "2d_image_search": {
        "path": "azureml/advanced_demos/image_search/2d_image_search.ipynb",
        "endpoints": ["MI2_MODEL_ENDPOINT"],
        "markers": ["mi2"],
    },
    "3d_image_search": {
        "path": "azureml/advanced_demos/image_search/3d_image_search.ipynb",
        "endpoints": ["MI2_MODEL_ENDPOINT"],
        "markers": ["mi2"],
    },
}


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            name,
            marks=[getattr(pytest.mark, m) for m in entry.get("markers", [])],
            id=name,
        )
        for name, entry in MANIFEST.items()
    ],
)
def test_notebook(name):
    entry = MANIFEST[name]
    skip_reason = entry.get("skip")
    if skip_reason:
        pytest.skip(skip_reason)

    endpoints = entry.get("endpoints", [])
    timeout = entry.get("timeout", DEFAULT_TIMEOUT)
    for env in endpoints:
        if not os.environ.get(env):
            pytest.skip(f"{env} not set")

    for rel_path in entry.get("requires_data", []):
        data_path = Path(DATA_ROOT) / rel_path
        if not data_path.exists():
            pytest.skip(f"required data not found: {data_path}")

    notebook = entry["path"]
    input_path = REPO_ROOT / notebook
    output_path = REPO_ROOT / ".papermill" / notebook
    output_path.parent.mkdir(parents=True, exist_ok=True)
    papermill.execute_notebook(
        str(input_path),
        str(output_path),
        kernel_name=KERNEL,
        cwd=str(input_path.parent),
        execution_timeout=timeout,
        log_output=True,
    )
