import argparse
import base64
import gzip
import json
from pathlib import Path

import mlflow
import nibabel as nib
import numpy as np
import pandas as pd


def decode_nifti(payload: str) -> np.ndarray:
    nifti_bytes = gzip.decompress(base64.b64decode(json.loads(payload)["data"]))
    image = nib.Nifti1Image.from_bytes(nifti_bytes)
    return np.asanyarray(image.dataobj)


def metrics(tp: int, fp: int, fn: int) -> dict:
    return {
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-column", default="nifti_file")
    args = parser.parse_args()

    cases = pd.read_csv(args.data / "cases.csv")
    classes = pd.read_csv(args.data / "classes.csv")
    predictions = pd.read_csv(args.predictions / "predictions.csv")
    rows = cases.merge(predictions, on="_row_id", validate="one_to_one")
    if len(rows) != len(cases):
        raise ValueError("Every evaluation case must have one prediction")

    confusion = {
        int(class_id): {"tp": 0, "fp": 0, "fn": 0} for class_id in classes["class_id"]
    }
    allowed_labels = [0, *confusion]
    for _, row in rows.iterrows():
        truth = np.asanyarray(nib.load(args.data / row["mask"]).dataobj)
        prediction = decode_nifti(row[args.prediction_column])
        if prediction.shape != truth.shape:
            raise ValueError(
                f"Prediction shape {prediction.shape} does not match truth shape "
                f"{truth.shape} for {row['case_id']}"
            )

        for name, labels in (("reference", truth), ("prediction", prediction)):
            unexpected = np.setdiff1d(labels, allowed_labels)
            if unexpected.size:
                raise ValueError(
                    f"Unexpected {name} labels {unexpected.tolist()} for "
                    f"{row['case_id']}; expected labels from {allowed_labels}"
                )

        for class_id in confusion:
            truth_positive = truth == class_id
            prediction_positive = prediction == class_id
            confusion[class_id]["tp"] += int(
                np.logical_and(truth_positive, prediction_positive).sum()
            )
            confusion[class_id]["fp"] += int(
                np.logical_and(~truth_positive, prediction_positive).sum()
            )
            confusion[class_id]["fn"] += int(
                np.logical_and(truth_positive, ~prediction_positive).sum()
            )

    class_names = dict(zip(classes["class_id"], classes["class_name"], strict=True))
    class_metrics = {
        str(class_id): {
            "class_name": class_names[class_id],
            **metrics(**confusion[class_id]),
        }
        for class_id in confusion
    }
    mlflow.log_metric("case_count", len(rows))
    for class_result in class_metrics.values():
        metric_prefix = "_".join(class_result["class_name"].lower().split())
        for metric_name in (
            "dice",
            "iou",
            "sensitivity",
            "precision",
            "tp",
            "fp",
            "fn",
        ):
            value = class_result[metric_name]
            if value is not None:
                mlflow.log_metric(f"{metric_prefix}_{metric_name}", value)

    macro_metrics = {}
    for metric_name in ("dice", "iou", "sensitivity", "precision"):
        values = [
            result[metric_name]
            for result in class_metrics.values()
            if result[metric_name] is not None
        ]
        macro_metrics[metric_name] = sum(values) / len(values) if values else None
        if macro_metrics[metric_name] is not None:
            mlflow.log_metric(f"macro_{metric_name}", macro_metrics[metric_name])

    result = {
        "case_count": len(rows),
        "class_metrics": class_metrics,
        "macro_metrics": macro_metrics,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metric_rows = [
        {
            "aggregation": "class",
            "class_id": int(class_id),
            "case_count": len(rows),
            **values,
        }
        for class_id, values in class_metrics.items()
    ]
    metric_rows.append(
        {"aggregation": "macro", "case_count": len(rows), **macro_metrics}
    )
    metrics_frame = pd.DataFrame(
        metric_rows,
        columns=[
            "aggregation",
            "class_id",
            "class_name",
            "case_count",
            "dice",
            "iou",
            "sensitivity",
            "precision",
            "tp",
            "fp",
            "fn",
        ],
    ).astype({column: "Int64" for column in ("class_id", "tp", "fp", "fn")})
    metrics_frame.to_csv(args.output / "metrics.csv", index=False)


if __name__ == "__main__":
    main()
