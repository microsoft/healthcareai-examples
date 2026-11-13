"""Invoke a named deployment with a prepared NIfTI volume and save its prediction."""

import argparse
import base64
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential
import nibabel as nib
import numpy as np
import pandas as pd


def predict_volume(
    ml_client: MLClient,
    *,
    endpoint_name: str,
    deployment_name: str,
    image_path: Path,
    text: str,
) -> nib.Nifti1Image:
    """Send an already-windowed volume; return the decoded prediction NIfTI."""
    request = {
        "input_data": {
            "columns": ["image", "text"],
            "index": [0],
            "data": [[base64.b64encode(image_path.read_bytes()).decode(), text]],
        }
    }
    with TemporaryDirectory(prefix="mip3d-request-") as directory:
        request_path = Path(directory) / "request.json"
        request_path.write_text(json.dumps(request))
        response = ml_client.online_endpoints.invoke(
            endpoint_name=endpoint_name,
            deployment_name=deployment_name,
            request_file=str(request_path),
        )
    encoded = json.loads(json.loads(response)[0]["nifti_file"])["data"]
    return nib.Nifti1Image.from_bytes(gzip.decompress(base64.b64decode(encoded)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-config", type=Path, default=Path("config.json"))
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--deployment-name", default="ich")
    parser.add_argument("--evaluation-data", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("results/prediction.nii.gz")
    )
    args = parser.parse_args()
    samples = pd.read_csv(args.evaluation_data / "data.csv")
    if not 0 <= args.sample_index < len(samples):
        parser.error("sample-index must identify a row in data.csv")
    sample = samples.iloc[args.sample_index]
    ml_client = MLClient.from_config(AzureCliCredential(), path=args.workspace_config)
    prediction = predict_volume(
        ml_client,
        endpoint_name=args.endpoint_name,
        deployment_name=args.deployment_name,
        image_path=args.evaluation_data / sample["image"],
        text=sample["text"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(prediction, args.output)
    labels = np.rint(np.asanyarray(prediction.dataobj)).astype(np.uint8)
    print("Prediction:", args.output)
    print("Prediction shape:", labels.shape)
    print("Predicted labels:", np.unique(labels))


if __name__ == "__main__":
    main()
