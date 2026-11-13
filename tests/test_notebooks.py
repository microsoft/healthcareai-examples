import ast
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

MANIFEST_opensource = {
    "zero-shot-classification": {
        "path": "azureml/medimageinsight/zero-shot-classification.ipynb",
        "endpoints": ["MI2_MODEL_ENDPOINT"],
        "markers": ["mi2"],
        "timeout": 1800,
        "parameters": {"variant": "opensource"},
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
        "parameters": {"variant": "opensource"},
    },
    "model-comparison": {
        "path": "azureml/medimageinsight/model-comparison.ipynb",
        "endpoints": ["MI2_MODEL_ENDPOINT"],
        "markers": ["mi2"],
        "timeout": 1800,
        "parameters": {"variant": "opensource"},
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
        "parameters": {"variant": "opensource"},
    },
    "adapter-training": {
        "path": "azureml/medimageinsight/adapter-training.ipynb",
        "endpoints": ["MI2_MODEL_ENDPOINT"],
        "markers": ["mi2"],
        "timeout": 1800,
        "parameters": {"variant": "opensource"},
        "depends_on": "zero-shot-classification-oss",
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
        "parameters": {"variant": "opensource"},
    },
    "rag_infection_detection": {
        "path": "azureml/advanced_demos/image_search/rag_infection_detection.ipynb",
        "endpoints": ["MI2_MODEL_ENDPOINT"],
        "markers": ["mi2"],
        "parameters": {"variant": "opensource"},
        # The WoundcareVQA dataset is not redistributed (licensing) and there is
        # no headless download for it yet, so skip unless the data is present.
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
        "parameters": {"variant": "opensource"},
    },
    "3d_image_search": {
        "path": "azureml/advanced_demos/image_search/3d_image_search.ipynb",
        "endpoints": ["MI2_MODEL_ENDPOINT"],
        "markers": ["mi2"],
        "parameters": {"variant": "opensource"},
    },
}


MANIFEST_premium = {
    "zero-shot-classification": {
        "path": "azureml/medimageinsight/zero-shot-classification.ipynb",
        "endpoints": ["MI2_PREMIUM_ENDPOINT"],
        "markers": ["mi2p"],
        "timeout": 1800,
        "parameters": {"variant": "premium"},
    },
    "adapter-training": {
        "path": "azureml/medimageinsight/adapter-training.ipynb",
        "endpoints": ["MI2_PREMIUM_ENDPOINT"],
        "markers": ["mi2p"],
        "timeout": 1800,
        "parameters": {"variant": "premium"},
        "depends_on": "zero-shot-classification-premium",
    },
}

MANIFEST = {
    **{name + "-oss": entry for name, entry in MANIFEST_opensource.items()},
    **{name + "-premium": entry for name, entry in MANIFEST_premium.items()},
}


@pytest.fixture(scope="session")
def run_notebook():
    completed = {}

    def run(name):
        entry = MANIFEST[name]
        notebook = entry["path"]
        if completed.get(notebook) == name:
            return
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

        dependency = entry.get("depends_on")
        if dependency:
            run(dependency)

        input_path = REPO_ROOT / notebook
        output_path = (REPO_ROOT / ".papermill" / notebook).with_name(f"{name}.ipynb")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        completed.pop(notebook, None)
        papermill.execute_notebook(
            str(input_path),
            str(output_path),
            parameters=entry.get("parameters", {}),
            kernel_name=KERNEL,
            cwd=str(input_path.parent),
            execution_timeout=timeout,
            log_output=True,
        )
        completed[notebook] = name

    return run


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
def test_notebook(name, run_notebook):
    run_notebook(name)


def test_mi2_case_parameters():
    from papermill.iorw import load_notebook_node
    from papermill.parameterize import parameterize_notebook

    for name, entry in MANIFEST.items():
        if not {"MI2_MODEL_ENDPOINT", "MI2_PREMIUM_ENDPOINT"}.intersection(
            entry.get("endpoints", [])
        ):
            continue
        notebook = load_notebook_node(str(REPO_ROOT / entry["path"]))
        if not any(
            cell.cell_type == "code" and "get_medimage_client(" in cell.source
            for cell in notebook.cells
        ):
            continue
        expected = "premium" if name.endswith("-premium") else "opensource"
        assert entry["parameters"]["variant"] == expected, name
        variant = None
        prepared = parameterize_notebook(notebook, entry["parameters"])
        for cell in prepared.cells:
            if cell.cell_type != "code":
                continue
            tree = ast.parse(cell.source)
            for statement in tree.body:
                if isinstance(statement, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "variant"
                    for target in statement.targets
                ):
                    variant = ast.literal_eval(statement.value)
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "get_medimage_client"
                for node in ast.walk(tree)
            ):
                assert variant == expected, name
