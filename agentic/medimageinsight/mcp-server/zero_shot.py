# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Zero-shot CLIP-style classification on MedImageInsight embeddings.

Splits the math out of the thin endpoint client. The ``MedImageInsightClient``
handles image reading, preprocessing, and endpoint calls; this module owns
text-embedding fetch, logit scaling, softmax conversion, and the bundled label
catalog used to discover example labels.
"""

import csv
import re
from pathlib import Path

import numpy as np

from errors import ToolRuntimeError
from mi2_client import MI2Client


def classify(
    file_path: str,
    labels: list[str],
    mi2_client: MI2Client | None = None,
) -> dict[str, float]:
    """Classify a medical image against text labels using MedImageInsight.
    Returns a softmax probability per label (scores sum to ~1, so labels
    compete — supply a mutually-exclusive label set).

    Labels follow ``"<modality> <body_part> <view> <condition>"`` format —
    e.g. ``"x-ray chest anteroposterior Pneumonia"``. Use
    ``label_examples()`` (or the ``zeroshot_label_examples`` MCP tool) to
    discover valid modality / body_part / view values.

    **Anti-patterns — avoid these label styles:**

    - ``"COVID-19"`` — too vague, less discriminative.
    - ``"chest X-ray showing COVID-19 pneumonia"`` — free-form prose skews
      probabilities vs. the structured format.

    **Tip:** Combine labels across modalities / body-parts / conditions in a
    single call to answer multiple questions at once. The top-scoring label
    tells you modality, anatomy, and condition together. E.g.::

        ["x-ray chest anteroposterior Pneumonia",
         "x-ray chest anteroposterior No Finding",
         "computed tomography chest axial Pneumonia",
         "magnetic resonance imaging brain sagittal normal"]

    Args:
        file_path: Path to the image file (PNG or JPEG).
        labels: Text labels to classify against, in the format above.
        mi2_client: Optional configured MedImageInsight client. If omitted,
            a client is configured from the environment.

    Returns:
        ``{label: probability}`` for each input label.
    """
    if mi2_client is None:
        raise ToolRuntimeError(
            "mi2_client is required. Configure the MCP server first or pass a client explicitly."
        )
    resolved_client = mi2_client
    img_result = resolved_client.submit(image_list=[file_path])
    img_feats = np.squeeze(np.array(img_result[0]["image_features"]))

    text_result = resolved_client.submit(text_list=labels)
    text_feats = np.stack(
        [np.squeeze(np.array(row["text_features"])) for row in text_result]
    )
    scaling = np.exp(float(text_result[0]["scaling_factor"]))
    logits = scaling * (img_feats @ text_feats.T)

    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    return dict(zip(labels, probs.tolist()))


def label_examples(
    group: str | None = None,
    regex: str | None = None,
) -> list[str]:
    """Look up built-in MedImageInsight label examples for use with
    ``zeroshot_classify`` / ``classify``.

    Use **one** of the two parameters per call:

    ``group``
        Return the unique values for a grouping column.
        Valid groups: ``modality``, ``body_part``, ``view``.

        Examples:

        - ``group="modality"``  → ``["x-ray", "ct", "mr", ...]``
        - ``group="body_part"`` → ``["abdomen", "brain", "chest", ...]``
        - ``group="view"``      → ``["anteroposterior", "axial", ...]``

    ``regex``
        Return label text strings whose text matches a regex
        (case-insensitive, Python ``re.search``).

        Examples:

        - ``regex="x-ray.*chest"`` — all chest X-ray labels
        - ``regex="pneumonia|effusion|atelectasis"`` — specific conditions
        - ``regex="pediatric.*hand"`` — bone-age labels

    Args:
        group: Return unique values for this column (mutually exclusive
            with *regex*).
        regex: Return label texts matching this pattern (mutually exclusive
            with *group*).

    Returns:
        A list of strings — either unique group values or matching label texts.
    """
    if group and regex:
        raise ToolRuntimeError("Specify group or regex, not both.")
    if not group and not regex:
        raise ToolRuntimeError("Specify either group or regex.")

    valid_groups = ("modality", "body_part", "view")
    csv_path = Path(__file__).with_name("label_examples.csv")
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if group:
        if group not in valid_groups:
            raise ToolRuntimeError(
                f"Invalid group {group!r}. Must be one of: {', '.join(valid_groups)}"
            )
        return sorted({row[group] for row in rows if row[group]})
    try:
        compiled = re.compile(regex, re.IGNORECASE)
    except re.error as exc:
        raise ToolRuntimeError(
            f"Invalid regex {regex!r}: {exc}. Provide a valid Python regular expression."
        ) from exc
    return [row["text"] for row in rows if compiled.search(row["text"])]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MI2 zero-shot classification (standalone, bypasses MCP)."
    )
    parser.add_argument("image", help="path to a medical image file (PNG or JPEG)")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=[
            "x-ray chest anteroposterior Pneumonia",
            "x-ray chest anteroposterior No Finding",
        ],
        help="text labels to classify against",
    )
    args = parser.parse_args()

    import os

    from dotenv import load_dotenv

    load_dotenv()
    endpoint = os.environ.get("MI2_MODEL_ENDPOINT", "").strip()
    if not endpoint:
        raise SystemExit("MI2_MODEL_ENDPOINT is not set.")
    result = classify(
        args.image, args.labels, mi2_client=MI2Client.from_endpoint(endpoint)
    )
    for label, prob in sorted(result.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{prob:.4f}  {label}")  # slopcop: ignore[no-print]
