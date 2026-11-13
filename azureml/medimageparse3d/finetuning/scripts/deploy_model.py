"""Reuse a named deployment or register and deploy a completed pipeline's model."""

import argparse
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
    OnlineRequestSettings,
    PipelineJob,
    ProbeSettings,
)
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import AzureCliCredential


def deploy_model(
    ml_client: MLClient,
    pipeline_job: PipelineJob,
    *,
    model_name: str,
    endpoint_name: str,
    deployment_name: str,
    instance_type: str,
) -> ManagedOnlineDeployment:
    """Reuse successful deployments unchanged; new deployments receive all traffic."""
    try:
        deployment = ml_client.online_deployments.get(
            name=deployment_name, endpoint_name=endpoint_name
        )
    except ResourceNotFoundError:
        deployment = None
    if deployment is not None:
        if deployment.provisioning_state != "Succeeded":
            raise RuntimeError(
                f"Deployment {endpoint_name}/{deployment_name} is "
                f"{deployment.provisioning_state}. Check its status before continuing."
            )
        print(f"Reusing deployment: {endpoint_name}/{deployment_name}")
        print("Deployed model:", deployment.model)
        return deployment
    if pipeline_job.status != "Completed":
        raise ValueError("The pipeline must be Completed before registering its model.")
    environment = pipeline_job.tags["inference_environment_asset_id"]
    registered_model = ml_client.models.create_or_update(
        Model(
            name=model_name,
            path=f"azureml://jobs/{pipeline_job.name}/outputs/mlflow_model_folder",
            type=AssetTypes.CUSTOM_MODEL,
            description="MedImageParse3D fine-tuned on BHSD.",
            tags={"inference_environment_asset_id": environment},
            properties={"training_job_id": pipeline_job.name},
        )
    )
    print(registered_model.id)
    try:
        endpoint = ml_client.online_endpoints.get(endpoint_name)
    except ResourceNotFoundError:
        endpoint = ManagedOnlineEndpoint(name=endpoint_name, auth_mode="key")
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    deployment = ml_client.online_deployments.begin_create_or_update(
        ManagedOnlineDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            model=registered_model.id,
            environment=environment,
            instance_type=instance_type,
            instance_count=1,
            request_settings=OnlineRequestSettings(request_timeout_ms=180000),
            liveness_probe=ProbeSettings(initial_delay=600),
        )
    ).result()
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    return deployment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-config", type=Path, default=Path("config.json"))
    parser.add_argument(
        "--job-name", required=True, help="Completed parent pipeline name."
    )
    parser.add_argument("--model-name", default="mip3d-bhsd")
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--deployment-name", default="ich")
    parser.add_argument("--instance-type", default="Standard_NC40ads_H100_v5")
    args = parser.parse_args()
    ml_client = MLClient.from_config(AzureCliCredential(), path=args.workspace_config)
    pipeline_job = ml_client.jobs.get(args.job_name)
    deploy_model(
        ml_client,
        pipeline_job,
        model_name=args.model_name,
        endpoint_name=args.endpoint_name,
        deployment_name=args.deployment_name,
        instance_type=args.instance_type,
    )


if __name__ == "__main__":
    main()
