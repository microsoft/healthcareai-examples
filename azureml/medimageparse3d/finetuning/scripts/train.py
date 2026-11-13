"""Submit a MedImageParse3D train, score, and evaluate pipeline."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from azure.ai.ml import Input, MLClient, dsl
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model, PipelineJob, UserIdentityConfiguration
from azure.identity import AzureCliCredential

from components.components import get_evaluate_component, get_score_component


def create_training_job(
    base_model: Model,
    finetune,
    score,
    evaluate,
    *,
    compute_name: str,
    experiment_name: str,
    training_data: str | Path,
    evaluation_data: str | Path,
    parameters_path: Path,
    job_name: str | None = None,
) -> PipelineJob:
    """Build the pipeline without submitting it or uploading data."""

    @dsl.pipeline(default_compute=compute_name)
    def training_pipeline(pretrained_model, training_data, test_data, config):
        trained = finetune(
            pretrained_mlflow_model=pretrained_model, data=training_data, config=config
        )
        scored = score(model=trained.outputs.mlflow_model_folder, data=test_data)
        evaluated = evaluate(predictions=scored.outputs.predictions, data=test_data)
        return {
            "mlflow_model_folder": trained.outputs.mlflow_model_folder,
            "predictions": scored.outputs.predictions,
            "metrics": evaluated.outputs.metrics,
            "training_output": trained.outputs.out,
        }

    job = training_pipeline(
        pretrained_model=Input(type=AssetTypes.URI_FOLDER, path=base_model.id),
        training_data=Input(
            type=AssetTypes.URI_FOLDER, path=str(training_data), mode="ro_mount"
        ),
        test_data=Input(
            type=AssetTypes.URI_FOLDER, path=str(evaluation_data), mode="ro_mount"
        ),
        config=Input(
            type=AssetTypes.URI_FILE, path=str(parameters_path), mode="ro_mount"
        ),
    )
    job.name = job_name or f"mip3d-bhsd-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    job.experiment_name = experiment_name
    job.tags = {
        "inference_environment_asset_id": base_model.tags[
            "inference_environment_asset_id"
        ]
    }
    job.identity = UserIdentityConfiguration()
    output_path = (
        "azureml://datastores/${{default_datastore}}/paths/"
        f"pipeline_outputs/{job.experiment_name}/"
        "${{name}}/${{output_name}}/"
    )
    for output in job.outputs.values():
        output.path = output_path
    return job


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-config", type=Path, default=Path("config.json"))
    parser.add_argument(
        "--training-data",
        required=True,
        help="Prepared training folder or Azure ML URI.",
    )
    parser.add_argument(
        "--evaluation-data",
        required=True,
        help="Prepared evaluation folder or Azure ML URI.",
    )
    parser.add_argument("--compute-name", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--job-name")
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "parameters.yaml",
    )
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Stream the parent pipeline through evaluation.",
    )
    args = parser.parse_args()
    credential = AzureCliCredential()
    ml_client = MLClient.from_config(credential, path=args.workspace_config)
    model_registry = MLClient(credential, registry_name="azureml-ai-frontiers")
    base_model = model_registry.models.get(name="MedImageParse3d", label="latest")
    component_registry = MLClient(credential, registry_name="azureml")
    finetune = component_registry.components.get(
        name="medimageparse_3d_finetune", version="0.0.2"
    )
    environment = base_model.tags["inference_environment_asset_id"]
    job = create_training_job(
        base_model,
        finetune,
        get_score_component(environment),
        get_evaluate_component(environment),
        compute_name=args.compute_name,
        experiment_name=args.experiment_name,
        training_data=args.training_data,
        evaluation_data=args.evaluation_data,
        parameters_path=args.parameters,
        job_name=args.job_name,
    )
    if args.force_rerun:
        job.settings.force_rerun = True
    submitted = ml_client.jobs.create_or_update(
        job, experiment_name=args.experiment_name
    )
    print("Job name:", submitted.name)
    print("Studio URL:", submitted.studio_url)
    if args.wait:
        ml_client.jobs.stream(submitted.name)


if __name__ == "__main__":
    main()
