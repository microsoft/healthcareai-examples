from pathlib import Path

from azure.ai.ml import Input, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Command

COMPONENT_ROOT = Path(__file__).resolve().parent


def get_score_component(environment: str) -> Command:
    return command(
        name="score_mlflow_model",
        code=COMPONENT_ROOT,
        command=(
            "python score.py --model ${{inputs.model}} --data ${{inputs.data}} "
            "--file-column image --output ${{outputs.predictions}}"
        ),
        environment=environment,
        inputs={
            "model": Input(type=AssetTypes.MLFLOW_MODEL),
            "data": Input(type=AssetTypes.URI_FOLDER),
        },
        outputs={"predictions": Output(type=AssetTypes.URI_FOLDER)},
    )


def get_evaluate_component(environment: str) -> Command:
    return command(
        name="evaluate_segmentation",
        code=COMPONENT_ROOT,
        command=(
            "python evaluate.py --predictions ${{inputs.predictions}} "
            "--data ${{inputs.data}} --output ${{outputs.metrics}}"
        ),
        environment=environment,
        inputs={
            "predictions": Input(type=AssetTypes.URI_FOLDER),
            "data": Input(type=AssetTypes.URI_FOLDER),
        },
        outputs={"metrics": Output(type=AssetTypes.URI_FOLDER)},
    )
