"""Convert paired medical volumes into slice-training and volume-scoring data.

Preparing slices: training input
-------------------------------
Pass <out> itself as the fine-tuning component's data input. The trainer expects::

        <out>/
        ├─ train/        RGB PNG, 512x512, uint8
        ├─ train_mask/   grayscale PNG, 512x512, uint8, pixel value == class ID
        ├─ train.json    {"class_prompts": {...}, "annotations": [...]}
        ├─ val/          optional; same image format as train/
        ├─ val_mask/     optional; same mask format as train_mask/
        └─ val.json      optional; same top-level fields as train.json

- RGB is the file format; channels contain [current, previous, next] slices.
- class_prompts maps string class IDs to lists of prompt phrases.
- Each annotation contains file_name, mask_file, split, class_ids, and
    instance_label. Filenames are relative to the corresponding image/mask folders.
- For precomputed validation, use split="val" in val.json and keep its patients
    separate from training. Call prepare_slices with split="val" and
    target_repeat={} in the same output root, then enable validation in the trainer.

Preparing volumes: scoring input
-------------------------------
prepare_volumes writes images/, masks/, data.csv, cases.csv, and classes.csv.
data.csv supplies image paths and prompts; cases.csv matches predictions to
reference masks by _row_id; classes.csv maps pixel IDs to names. This separate
folder is used after training, not as the trainer's optional validation input.

Transforms
----------
- Image and label must already share a voxel grid. Reorienting to canonical axes
    changes array layout, not anatomical alignment; it is not registration.
- Move the slice axis first for training and pack [current, previous, next] as
    channels. The mask labels only the current slice. At volume boundaries, the
    adjacent interior slice supplies the missing neighbor.
- Apply one fixed intensity window and scale images to uint8 [0, 255]. Use the
    same transform for training and scoring; do not independently stretch slices.
- Preserve integer class IDs in masks. If adapting to other image sizes, match
    serving geometry and use nearest-neighbor interpolation for masks. Padding
    and resizing are not implemented here.
- Keep scoring images as complete volumes. Reference masks use the prediction's
    array order with identity affines for voxel comparison, not spatial display.
    Verify that order for another dataset or model rather than assuming an axis.

Separate conversion from training policy: split patients before selecting or
repeating slices, and never oversample the held-out data. This example's negative
sampling and repetition rules are recipe choices, not format requirements.
The notebook selects pairs and partitions, applies exclusions, and supplies class
names and prompts. See the notebook for those choices and validation.

Both helpers accept {case_id: {"image": path, "gt": path}}. Case IDs become
output filename stems. Training accepts NIfTI, MHA, or NRRD volumes;
scoring preparation expects NIfTI images and ground-truth masks.
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from numpy.typing import DTypeLike
from tqdm.auto import tqdm


def load_volume(path: Path | str, dtype: DTypeLike) -> np.ndarray:
    """Read canonical slice-first (Z, X, Y) pixels with the requested dtype."""
    path = Path(path)
    if path.name.endswith((".nii", ".nii.gz")):
        volume = nib.as_closest_canonical(nib.load(path))
        return np.asarray(volume.dataobj, dtype=dtype).transpose(2, 0, 1)
    volume = sitk.DICOMOrient(sitk.ReadImage(str(path)), "RAS")
    return np.asarray(sitk.GetArrayFromImage(volume), dtype=dtype).transpose(0, 2, 1)


def _validate_window(width: float, level: float) -> None:
    if not np.isfinite(width) or width <= 0:
        raise ValueError("window_width must be finite and greater than zero")
    if not np.isfinite(level):
        raise ValueError("window_level must be finite")


def window_ct(image: np.ndarray, width: float, level: float) -> np.ndarray:
    """Use the same HU window for training and scoring; never window class masks."""
    _validate_window(width, level)
    lower = level - width / 2
    return ((np.clip(image, lower, lower + width) - lower) / width * 255).astype(
        np.uint8
    )


def prepare_slices(
    cases: dict[str, dict[str, Path | str]],
    output_dir: Path,
    *,
    class_prompts: dict[str, list[str]],
    window_width: float,
    window_level: float,
    target_repeat: dict[int, int],
    negative_sample_ratio: float,
    seed: int,
    split: str = "train",
) -> int:
    """Write train or val slices using the layout in Preparing slices above.

    split controls the image folder, mask folder, manifest, and annotation labels.
    Use separate held-out cases and target_repeat={} for split="val".
    Input pairs must already be aligned; only array shapes are checked here.
    target_repeat maps class IDs to total copies per slice. Use the largest
    matching count; unlisted classes and background slices get one copy.
    negative_sample_ratio sets negatives per foreground slice in each volume,
    rounded and capped by available negatives. Use zero to omit negatives.
    Repeatable negative selection depends on BOTH cases order and seed: one
    RNG seeded with seed + 1 spans all cases. The return count includes repeated
    manifest entries, not just unique PNGs. No validation split is performed here.
    """
    if split not in ("train", "val"):
        raise ValueError("split must be 'train' or 'val'")
    _validate_window(window_width, window_level)
    if not np.isfinite(negative_sample_ratio) or negative_sample_ratio < 0:
        raise ValueError("negative_sample_ratio must be finite and nonnegative")
    if any(
        isinstance(count, (bool, np.bool_))
        or not isinstance(count, (int, np.integer))
        or count < 1
        for count in target_repeat.values()
    ):
        raise ValueError("target_repeat counts must be positive integers")
    image_dir = output_dir / split
    mask_dir = output_dir / f"{split}_mask"
    for directory in (image_dir, mask_dir):
        directory.mkdir(parents=True)

    annotations = []
    sampling_rng = random.Random(seed + 1)
    for case_id, paths in tqdm(cases.items(), desc=f"{split} volumes", unit="volume"):
        image = load_volume(paths["image"], dtype=np.float32)
        truth = load_volume(paths["gt"], dtype=np.uint8)
        if image.ndim != 3 or truth.ndim != 3:
            raise ValueError(f"Expected scalar 3D volumes: {case_id}")
        if image.shape != truth.shape:
            raise ValueError(
                f"Image and ground truth must have matching shapes: {case_id}"
            )
        source = window_ct(image, window_width, window_level)
        source = np.stack(
            (
                source,
                np.concatenate((source[1:2], source[:-1]), axis=0),
                np.concatenate((source[1:], source[-2:-1]), axis=0),
            ),
            axis=-1,
        )
        # Keep all labeled slices and sample negatives relative to that count.
        has_foreground = truth.any(axis=(1, 2))
        foreground = np.flatnonzero(has_foreground).tolist()
        background = np.flatnonzero(~has_foreground).tolist()
        sampling_rng.shuffle(background)
        selected = sorted(
            foreground + background[: round(len(foreground) * negative_sample_ratio)]
        )
        # Write each slice once; oversample configured classes in the manifest.
        for slice_index in selected:
            file_name = f"{case_id}_{slice_index:03d}.png"
            cv2.imwrite(str(image_dir / file_name), source[slice_index])
            cv2.imwrite(str(mask_dir / file_name), truth[slice_index])
            class_ids = [
                int(value) for value in np.unique(truth[slice_index]) if value != 0
            ]
            annotation = {
                "file_name": file_name,
                "mask_file": file_name,
                "split": split,
                "class_ids": class_ids,
                "instance_label": 0,
            }
            copies = max(
                (target_repeat.get(class_id, 1) for class_id in class_ids), default=1
            )
            annotations.extend([annotation] * copies)

    (output_dir / f"{split}.json").write_text(
        json.dumps(
            {"class_prompts": class_prompts, "annotations": annotations}, indent=2
        )
        + "\n"
    )
    return len(annotations)


def prepare_volumes(
    cases: dict[str, dict[str, Path | str]],
    output_dir: Path,
    *,
    class_names: dict[int, str],
    class_prompts: dict[str, list[str]],
    window_width: float,
    window_level: float,
) -> int:
    """Prepare NIfTI scoring volumes with prompts ordered by class_names.

    Reference masks have identity affines for array comparison, not display.
    Verify their axis order against model output before adapting another dataset.
    """
    _validate_window(window_width, window_level)
    for directory in (output_dir / "images", output_dir / "masks"):
        directory.mkdir(parents=True)

    # Prepare complete test volumes for scoring
    prompt = "&".join(class_prompts[str(class_id)][0] for class_id in class_names)
    score_rows = []
    case_rows = []
    for row_id, (case_id, paths) in enumerate(
        tqdm(cases.items(), desc="Scoring volumes", unit="volume")
    ):
        source = nib.load(paths["image"])
        truth = nib.load(paths["gt"])
        if len(source.shape) != 3 or len(truth.shape) != 3:
            raise ValueError(f"Expected scalar 3D NIfTI volumes: {case_id}")
        if source.shape != truth.shape or not np.allclose(
            source.affine, truth.affine, atol=1e-4, rtol=0
        ):
            raise ValueError(
                f"Image and ground truth must share a voxel grid: {case_id}"
            )
        image = window_ct(
            np.asarray(source.dataobj, dtype=np.float32), window_width, window_level
        )
        header = source.header.copy()
        header.set_data_dtype(np.uint8)
        file_name = f"{case_id}.nii.gz"
        nib.save(
            nib.Nifti1Image(image, source.affine, header),
            output_dir / "images" / file_name,
        )
        mask = np.asarray(truth.dataobj, dtype=np.uint8)
        nib.save(
            nib.Nifti1Image(np.moveaxis(mask, 2, 0), np.eye(4)),
            output_dir / "masks" / file_name,
        )
        score_rows.append(
            {"_row_id": row_id, "image": f"images/{file_name}", "text": prompt}
        )
        case_rows.append(
            {"_row_id": row_id, "case_id": case_id, "mask": f"masks/{file_name}"}
        )

    pd.DataFrame(score_rows).to_csv(output_dir / "data.csv", index=False)
    pd.DataFrame(case_rows).to_csv(output_dir / "cases.csv", index=False)
    pd.DataFrame(class_names.items(), columns=("class_id", "class_name")).to_csv(
        output_dir / "classes.csv", index=False
    )
    return len(score_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare slice or volume inputs from explicit image/mask pairs."
    )
    parser.add_argument("mode", choices=("slices", "volumes"))
    parser.add_argument(
        "--cases",
        type=Path,
        required=True,
        help="JSON case_id -> image/gt path mapping.",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        required=True,
        help="JSON containing class_names and class_prompts.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-width", type=float, default=200)
    parser.add_argument("--window-level", type=float, default=40)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument(
        "--target-repeat",
        default="{}",
        help='JSON class IDs to copies, e.g. \'{"1": 5, "5": 5}\'.',
    )
    parser.add_argument("--negative-sample-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    cases = {
        case_id: {key: args.cases.parent / Path(paths[key]) for key in ("image", "gt")}
        for case_id, paths in cases.items()
    }
    classes = json.loads(args.classes.read_text())
    options = dict(
        class_prompts=classes["class_prompts"],
        window_width=args.window_width,
        window_level=args.window_level,
    )
    if args.mode == "slices":
        target_repeat = {
            int(key): value for key, value in json.loads(args.target_repeat).items()
        }
        count = prepare_slices(
            cases,
            args.output_dir,
            split=args.split,
            target_repeat=target_repeat,
            negative_sample_ratio=args.negative_sample_ratio,
            seed=args.seed,
            **options,
        )
    else:
        count = prepare_volumes(
            cases,
            args.output_dir,
            class_names={
                int(key): value for key, value in classes["class_names"].items()
            },
            **options,
        )
    print(f"Prepared {count} records from {len(cases)} volumes in {args.output_dir}")


if __name__ == "__main__":
    main()
