"""Download evaluation results for the MedImageParse3D example pipeline."""

import argparse
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import PipelineJob
from azure.identity import AzureCliCredential
from azure.storage.blob import BlobClient


def download_metrics(
    ml_client: MLClient, pipeline_job: PipelineJob, output_dir: Path
) -> Path:
    """Download metrics.csv using the existing az login session, without prompts.

    Uses this example's pipeline_outputs/<experiment>/<evaluated-step>/metrics/
    layout in the default datastore. The signed-in user needs blob read access.
    """
    if pipeline_job.status != "Completed":
        raise ValueError("The pipeline must be Completed before downloading metrics.")
    evaluated_job = next(
        (
            child
            for child in ml_client.jobs.list(parent_job_name=pipeline_job.name)
            if child.display_name == "evaluated"
        ),
        None,
    )
    if evaluated_job is None:
        raise ValueError(f"No evaluated step found for pipeline {pipeline_job.name}.")

    datastore = ml_client.datastores.get_default()
    blob_path = (
        f"pipeline_outputs/{pipeline_job.experiment_name}/"
        f"{evaluated_job.name}/metrics/metrics.csv"
    )
    with BlobClient(
        account_url=f"https://{datastore.account_name}.blob.{datastore.endpoint}",
        container_name=datastore.container_name,
        blob_name=blob_path,
        credential=AzureCliCredential(),
    ) as blob:
        contents = blob.download_blob().readall()

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    metrics_path.write_bytes(contents)
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-config", type=Path, default=Path("config.json"))
    parser.add_argument(
        "--job-name", required=True, help="Completed parent pipeline name."
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    ml_client = MLClient.from_config(AzureCliCredential(), path=args.workspace_config)
    pipeline_job = ml_client.jobs.get(args.job_name)
    output_dir = args.output_dir or Path("results") / args.job_name
    print(download_metrics(ml_client, pipeline_job, output_dir))


if __name__ == "__main__":
    main()
