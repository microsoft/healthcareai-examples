# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Adapter classification on top of MedImageInsight embeddings.

Loads a pickled sklearn classifier trained with
``azureml/medimageinsight/adapter-training.ipynb`` and applies it to a
fresh image embedding from MI2.

The adapter bundle (``models/adapter_<head>/weights.joblib``) carries:

- ``model``              — fitted sklearn estimator with ``predict_proba``
- ``labels``             — class names in column order of ``predict_proba``
- ``operating_points``   — per-label thresholds: ``youden_j`` (rule-in) and
                           ``sens90`` (rule-out)

Adapter bundles are shipped with this example under
``models/adapter_<head>/weights.joblib``. To regenerate them, use the
adapter-training notebook (``azureml/medimageinsight/adapter-training.ipynb``)
and write the output to the same path.
"""

from pathlib import Path

import joblib
import numpy as np

from errors import ToolRuntimeError
from mi2_client import MI2Client

# Resolves to <server-dir>/models/
_MODELS_DIR = Path(__file__).resolve().parent / "models"

_VALID_HEADS = ("svm", "mlp")


def _load_bundle(head: str) -> dict:
    if head not in _VALID_HEADS:
        raise ToolRuntimeError(f"head must be one of {_VALID_HEADS} (got {head!r})")
    bundle_path = _MODELS_DIR / f"adapter_{head}" / "weights.joblib"
    if not bundle_path.exists():
        raise ToolRuntimeError(
            f"Adapter bundle not found: {bundle_path}\n"
            "Generate it using azureml/medimageinsight/adapter-training.ipynb "
            "and place the output at the path above."
        )
    return joblib.load(bundle_path)


def classify(
    file_path: str,
    head: str = "mlp",
    mi2_client: MI2Client | None = None,
) -> dict[str, str]:
    """Classify a chest X-ray with a trained adapter on top of MedImageInsight.

    Loads the requested adapter (``models/adapter_<head>/weights.joblib``),
    embeds the image with MI2, and returns a per-label call:

    - ``"positive"`` — score ≥ Youden-J threshold (rule-in).
    - ``"negative"`` — score ≤ sens90 threshold (rule-out, ≥0.90 sensitivity).
    - ``"possible"`` — score between the two (cannot rule in or out).

    Both thresholds are fit per label during training and stored in the
    adapter bundle. If no bundle is available, see ``models/README.md`` for
    instructions on generating one.

    Args:
        file_path: Path to the image file (PNG or JPEG).
        head: Which trained head to use — ``"mlp"`` (default) or ``"svm"``.
        mi2_client: Optional configured MedImageInsight client. If omitted,
            a client is configured from the environment.

    Returns:
        Dictionary mapping each label to ``"positive"``, ``"possible"``,
        or ``"negative"``.
    """
    bundle = _load_bundle(head)
    labels = list(bundle["labels"])
    op_points = bundle.get("operating_points") or {}

    if mi2_client is None:
        raise ToolRuntimeError(
            "mi2_client is required. Configure the MCP server first or pass a client explicitly."
        )
    resolved_client = mi2_client
    img_result = resolved_client.submit(image_list=[file_path])
    feats = np.asarray(img_result[0]["image_features"], dtype=np.float32).reshape(1, -1)

    proba = bundle["model"].predict_proba(feats)[0]
    out: dict[str, str] = {}
    for lbl, p in zip(labels, proba):
        op = op_points.get(lbl, {})
        rule_in = float(op.get("youden_j", 0.5))
        rule_out = float(op.get("sens90", 0.0))
        score = float(p)
        if score >= rule_in:
            out[lbl] = "positive"
        elif score <= rule_out:
            out[lbl] = "negative"
        else:
            out[lbl] = "possible"
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MI2 trained adapter classification (standalone, bypasses MCP)."
    )
    parser.add_argument(
        "image",
        help="path to a chest X-ray image (e.g., samples/00026132_011.png)",
    )
    parser.add_argument(
        "--head",
        default="mlp",
        choices=["mlp", "svm"],
        help="adapter head to use (default: mlp)",
    )
    args = parser.parse_args()

    import os

    from dotenv import load_dotenv

    load_dotenv()
    endpoint = os.environ.get("MI2_MODEL_ENDPOINT", "").strip()
    if not endpoint:
        raise SystemExit("MI2_MODEL_ENDPOINT is not set.")
    result = classify(
        args.image, head=args.head, mi2_client=MI2Client.from_endpoint(endpoint)
    )
    for label, bucket in result.items():
        print(f"{label:30s} {bucket}")  # slopcop: ignore[no-print]
