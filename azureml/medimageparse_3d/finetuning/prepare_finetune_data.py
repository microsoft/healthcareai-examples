"""Prepare NIfTI volumes + label maps for MedImageParse3D (BiomedParse v2) fine-tuning.

The fine-tuning pipeline trains the *2D* ``BiomedParseModel`` on pre-sliced 2.5D RGB
PNGs, so volumes must be converted offline into the exact layout the trainer's
``BiomedSegDataset`` expects::

    <out>/
    |-- train/        RGB PNG, 512x512, uint8
    |-- train_mask/   grayscale PNG, 512x512, uint8, pixel value == class id
    `-- train.json    {"class_prompts": {...}, "annotations": [...]}

Every preprocessing step here mirrors inference (``utils.process_input`` and
``BiomedParseModel3D.forward_eval``) so that fine-tuning does not introduce a
train/serve skew:

* slice axis chosen by the same shape heuristic as ``utils.get_axis``
* centre zero-pad to a square, then resize to 512 (bicubic image / nearest mask)
* intensities rescaled to 0-255 (the model normalises with a scalar mean/std)
* channels packed as ``ch0 = slice d``, ``ch1 = slice d-1``, ``ch2 = slice d+1``
  (verified byte-identical against the shipped CVPR-MR-crossmoda-sample)

Example
-------
::

    python prepare_finetune_data.py \
        --images-dir  raw/images \
        --labels-dir  raw/labels \
        --out         MY_DATASET \
        --classes     '{"1": "vestibular schwannoma", "2": "cochlea"}' \
        --modality    MR \
        --site        brain

Validate a folder that was prepared some other way::

    python prepare_finetune_data.py --validate-only MY_DATASET
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:  # Prefer torch so the resize matches inference bit-for-bit.
    import torch
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised only on torch-less machines
    _HAS_TORCH = False


IMAGE_SUFFIXES = (".nii", ".nii.gz", ".mha", ".mhd", ".nrrd")

# Mirrors the paraphrase style of the shipped crossmoda sample. One is drawn at
# random per sample during training, so variety here is what makes the tuned
# model robust to prompt wording at inference time.
PROMPT_TEMPLATES = (
    "{name}",
    "{name} in {site} {modality}",
    "{modality} imaging of {name} in {site}",
    "Visualization of {name} in {site} {modality}",
    "{name} observed in {site} {modality}",
    "Segmentation of {name} in {site} {modality}",
    "Delineation of {name} in {site} {modality} imaging",
    "Identification of {name} in {site} {modality}",
    "Localization of {name} in {site} {modality}",
    "Characterization of {name} in {site} {modality}",
)


# --------------------------------------------------------------------------- #
# Geometry - kept byte-compatible with BiomedParse/utils.py
# --------------------------------------------------------------------------- #
def get_axis(shape) -> int:
    """Pick the slice axis exactly like ``utils.get_axis`` does at inference."""
    diff_ratio = [
        2 * abs(shape[1] - shape[2]) / (shape[1] + shape[2]),
        2 * abs(shape[0] - shape[2]) / (shape[0] + shape[2]),
        2 * abs(shape[0] - shape[1]) / (shape[0] + shape[1]),
    ]
    if diff_ratio[0] < 0.5:
        return 0
    return int(np.argmin(shape))


def get_padding(vol: np.ndarray):
    """Return the centred pad widths that make each slice square."""
    shape = vol.shape[1:]
    if shape[0] > shape[1]:
        pad1 = (shape[0] - shape[1]) // 2
        pad2 = (shape[0] - shape[1]) - pad1
        return [[0, 0], [0, 0], [pad1, pad2]]
    pad1 = (shape[1] - shape[0]) // 2
    pad2 = (shape[1] - shape[0]) - pad1
    return [[0, 0], [pad1, pad2], [0, 0]]


def resize_volume(vol: np.ndarray, size: int, *, is_mask: bool) -> np.ndarray:
    """Resize a (D, H, W) volume to (D, size, size)."""
    if is_mask:
        return np.stack(
            [
                cv2.resize(sl, (size, size), interpolation=cv2.INTER_NEAREST)
                for sl in vol.astype(np.uint8)
            ]
        )
    if _HAS_TORCH:
        tensor = torch.from_numpy(vol.astype(np.float32)).unsqueeze(0)
        resized = F.interpolate(
            tensor, size=(size, size), mode="bicubic", align_corners=False
        )
        return resized.squeeze(0).numpy()
    return np.stack(
        [
            cv2.resize(sl, (size, size), interpolation=cv2.INTER_CUBIC)
            for sl in vol.astype(np.float32)
        ]
    )


def pad_and_resize(vol: np.ndarray, size: int, *, is_mask: bool) -> np.ndarray:
    """Zero-pad each slice to a square then resize, as ``process_input`` does."""
    pad_width = get_padding(vol)
    vol = np.pad(vol, pad_width, mode="constant", constant_values=0)
    return resize_volume(vol, size, is_mask=is_mask)


# --------------------------------------------------------------------------- #
# Intensity
# --------------------------------------------------------------------------- #
def scale_intensity(vol: np.ndarray, ct_window: tuple[float, float] | None) -> np.ndarray:
    """Rescale a volume to 0-255 floats.

    The model normalises with a single scalar mean/std (``gray_scale=True``), so
    the absolute 0-255 range matters far more than per-slice contrast. Statistics
    are taken over the whole volume to keep neighbouring slices consistent.
    """
    vol = vol.astype(np.float32)
    if ct_window is not None:
        level, width = ct_window
        lo, hi = level - width / 2.0, level + width / 2.0
    else:
        lo, hi = np.percentile(vol, 0.5), np.percentile(vol, 99.5)
    if hi <= lo:
        lo, hi = float(vol.min()), float(vol.max())
    if hi <= lo:
        return np.zeros_like(vol)
    return (np.clip(vol, lo, hi) - lo) / (hi - lo) * 255.0


def pack_rgb(vol: np.ndarray) -> np.ndarray:
    """Pack a (D, H, W) volume into (D, H, W, 3) 2.5D slices.

    Reproduces ``BiomedParseModel3D.forward_eval``: ``ch0`` is the current slice,
    ``ch1`` the previous one and ``ch2`` the next one, with the first/last slice
    borrowing their neighbour's neighbour.
    """
    if vol.shape[0] == 1:
        return np.repeat(vol[..., None], 3, axis=-1)
    prev = np.concatenate((vol[1:2], vol[:-1]), axis=0)
    nxt = np.concatenate((vol[1:], vol[-2:-1]), axis=0)
    return np.stack((vol, prev, nxt), axis=-1)


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
def build_class_prompts(classes: dict[str, str], modality: str, site: str) -> dict:
    """Expand ``{class_id: name}`` into the paraphrase lists stored in the JSON."""
    prompts = {}
    for class_id, name in classes.items():
        rendered = [
            tpl.format(name=name, modality=modality, site=site)
            for tpl in PROMPT_TEMPLATES
        ]
        # Preserve order while removing duplicates (e.g. when site/modality blank).
        prompts[str(int(class_id))] = list(dict.fromkeys(rendered))
    return prompts


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #
@dataclass
class Case:
    """A single image volume paired with its label volume."""

    stem: str
    image: Path
    label: Path


def load_volume(path: Path) -> np.ndarray:
    """Load a medical volume as a numpy array in canonical (RAS) orientation."""
    if path.name.endswith((".nii", ".nii.gz")):
        import nibabel as nib

        img = nib.as_closest_canonical(nib.load(str(path)))
        return np.asanyarray(img.dataobj)

    import SimpleITK as sitk

    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def strip_suffix(name: str) -> str:
    """Remove a medical-image extension from a file name."""
    for suffix in sorted(IMAGE_SUFFIXES, key=len, reverse=True):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def discover_cases(images_dir: Path, labels_dir: Path, label_suffix: str) -> list[Case]:
    """Pair every image with its label by matching file stems."""
    cases: list[Case] = []
    for image in sorted(images_dir.iterdir()):
        if not image.name.endswith(IMAGE_SUFFIXES):
            continue
        stem = strip_suffix(image.name)
        candidates = [
            labels_dir / f"{stem}{label_suffix}{ext}" for ext in IMAGE_SUFFIXES
        ]
        label = next((c for c in candidates if c.exists()), None)
        if label is None:
            print(f"  ! no label for {image.name}, skipping", file=sys.stderr)
            continue
        cases.append(Case(stem=stem, image=image, label=label))
    return cases


def convert(args: argparse.Namespace) -> int:
    """Convert every discovered case into slices and write ``train.json``."""
    rng = random.Random(args.seed)
    out = Path(args.out)
    split = args.split
    img_dir = out / split
    mask_dir = out / f"{split}_mask"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    classes = json.loads(args.classes)
    class_prompts = build_class_prompts(classes, args.modality, args.site)
    known_ids = {int(k) for k in class_prompts}
    if max(known_ids) > 255:
        raise SystemExit("class ids must be <= 255 to fit in an 8-bit mask PNG")

    cases = discover_cases(Path(args.images_dir), Path(args.labels_dir), args.label_suffix)
    if not cases:
        raise SystemExit("no image/label pairs found")
    print(f"Found {len(cases)} case(s)")

    ct_window = None
    if args.ct_window:
        level, width = (float(v) for v in args.ct_window.split(","))
        ct_window = (level, width)

    annotations = []
    n_pos = n_neg = 0

    for case in cases:
        image = load_volume(case.image).astype(np.float32)
        label = load_volume(case.label)

        if image.shape != label.shape:
            print(
                f"  ! {case.stem}: image {image.shape} != label {label.shape}, skipping",
                file=sys.stderr,
            )
            continue

        axis = get_axis(image.shape)
        image = np.moveaxis(image, axis, 0)
        label = np.moveaxis(label, axis, 0)

        unexpected = set(np.unique(label)) - {0} - known_ids
        if unexpected:
            print(
                f"  ! {case.stem}: label values {sorted(unexpected)} are absent from "
                f"--classes and will be dropped",
                file=sys.stderr,
            )
            label = np.where(np.isin(label, list(known_ids)), label, 0)

        image = scale_intensity(image, ct_window)
        image = pad_and_resize(image, args.size, is_mask=False)
        label = pad_and_resize(label, args.size, is_mask=True)
        rgb = np.clip(pack_rgb(image), 0, 255).astype(np.uint8)

        foreground = [d for d in range(label.shape[0]) if label[d].any()]
        empty = [d for d in range(label.shape[0]) if not label[d].any()]
        rng.shuffle(empty)
        negatives = empty[: int(round(len(foreground) * args.negative_ratio))]
        keep = sorted(foreground + negatives)

        if not keep:
            print(f"  ! {case.stem}: no labelled slices, skipping", file=sys.stderr)
            continue

        for d in keep:
            fname = f"{args.prefix}{case.stem}_{d:03d}.png"
            # cv2 round-trips channel order, so ch0/ch1/ch2 survive as written.
            cv2.imwrite(str(img_dir / fname), rgb[d])
            cv2.imwrite(str(mask_dir / fname), label[d].astype(np.uint8))
            present = sorted(int(v) for v in np.unique(label[d]) if v != 0)
            annotations.append(
                {
                    "mask_file": fname,
                    "file_name": fname,
                    "split": split,
                    "class_ids": present,
                    "instance_label": 1 if args.instance_label else 0,
                }
            )
            n_pos += bool(present)
            n_neg += not present

        print(
            f"  {case.stem}: axis={axis} slices={image.shape[0]} "
            f"kept={len(keep)} (+{len(negatives)} empty)"
        )

    payload = {"class_prompts": class_prompts, "annotations": annotations}
    with open(out / f"{split}.json", "w") as handle:
        json.dump(payload, handle)

    print(f"\nWrote {len(annotations)} slices ({n_pos} with foreground, {n_neg} empty)")
    print(f"Dataset root: {out.resolve()}")
    return validate(out, split)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate(root: Path, split: str = "train") -> int:
    """Check a prepared folder against everything the trainer silently tolerates."""
    print(f"\nValidating {root} ...")
    errors: list[str] = []
    warnings: list[str] = []

    json_path = root / f"{split}.json"
    if not json_path.exists():
        print(f"FAIL: {json_path} is missing")
        return 1

    with open(json_path) as handle:
        payload = json.load(handle)

    class_prompts = payload.get("class_prompts")
    annotations = payload.get("annotations", [])
    if not annotations:
        errors.append("'annotations' is empty")
    if not class_prompts:
        warnings.append(
            "no top-level 'class_prompts'; each annotation must carry its own"
        )
    else:
        known = {int(k) for k in class_prompts if k != "instance_label"}
        for key, value in class_prompts.items():
            if key == "instance_label":
                continue
            if not isinstance(value, list) or not value:
                errors.append(f"class_prompts['{key}'] must be a non-empty list")
            elif len(value) < 3:
                warnings.append(
                    f"class_prompts['{key}'] has only {len(value)} paraphrase(s); "
                    "5-10 improves prompt robustness"
                )

    seen_values: set[int] = set()
    for ann in annotations:
        image_path = root / split / ann["file_name"]
        mask_path = root / f"{split}_mask" / ann["mask_file"]

        if not image_path.exists():
            errors.append(f"missing image {image_path.name}")
            continue
        if not mask_path.exists():
            errors.append(f"missing mask {mask_path.name}")
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

        if image is None or mask is None:
            errors.append(f"unreadable pair {ann['file_name']}")
            continue
        if image.ndim != 3 or image.shape[2] != 3:
            errors.append(f"{image_path.name}: image must be 3-channel RGB")
        if mask.ndim != 2:
            errors.append(
                f"{mask_path.name}: mask must be single-channel grayscale, got "
                f"shape {mask.shape}"
            )
            continue
        if image.shape[:2] != mask.shape[:2]:
            errors.append(f"{mask_path.name}: image/mask size mismatch")

        values = np.unique(mask)
        seen_values.update(int(v) for v in values)
        # The dataset zeroes out BOTH image and mask for {0, 255} masks, so a
        # binary export trains on nothing at all while looking perfectly healthy.
        if mask.max() == 255 and len(values) == 2:
            errors.append(
                f"{mask_path.name}: binary {{0, 255}} mask - the dataset will "
                "silently zero this sample; encode pixels as class ids"
            )

    if class_prompts:
        stray = seen_values - {0} - known
        if stray:
            errors.append(
                f"mask values {sorted(stray)} have no entry in class_prompts"
            )
        unused = known - seen_values
        if unused:
            warnings.append(f"class ids {sorted(unused)} never appear in any mask")

    for warning in warnings:
        print(f"  WARN: {warning}")
    for error in errors[:20]:
        print(f"  FAIL: {error}")
    if len(errors) > 20:
        print(f"  ... and {len(errors) - 20} more")

    if errors:
        print(f"\nInvalid: {len(errors)} error(s)")
        return 1
    print(f"Valid: {len(annotations)} slices, class ids {sorted(seen_values - {0})}")
    return 0


def main() -> int:
    """Parse arguments and dispatch to conversion or validation."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--validate-only",
        metavar="DATASET_DIR",
        help="validate an already-prepared dataset folder and exit",
    )
    parser.add_argument("--images-dir", help="folder of image volumes")
    parser.add_argument("--labels-dir", help="folder of label volumes")
    parser.add_argument("--out", help="output dataset folder")
    parser.add_argument(
        "--classes",
        help='JSON mapping of class id to target name, e.g. \'{"1": "tumor"}\'',
    )
    parser.add_argument("--modality", default="", help='e.g. "MR", "CT"')
    parser.add_argument("--site", default="", help='e.g. "brain", "abdomen"')
    parser.add_argument("--split", default="train", help="split name (default: train)")
    parser.add_argument("--size", type=int, default=512, help="output size")
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=0.15,
        help="empty slices to keep, as a fraction of foreground slices",
    )
    parser.add_argument(
        "--ct-window",
        help='CT window as "level,width", e.g. "40,400"; omit for percentile scaling',
    )
    parser.add_argument("--label-suffix", default="", help='e.g. "_seg"')
    parser.add_argument("--prefix", default="", help="prefix for output file names")
    parser.add_argument(
        "--instance-label",
        action="store_true",
        help="masks hold instance ids rather than class ids",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.validate_only:
        return validate(Path(args.validate_only), args.split)

    missing = [
        name
        for name in ("images_dir", "labels_dir", "out", "classes")
        if not getattr(args, name)
    ]
    if missing:
        parser.error("missing required arguments: " + ", ".join(f"--{m.replace('_', '-')}" for m in missing))

    if not _HAS_TORCH:
        print(
            "note: torch not installed, falling back to cv2 bicubic resize "
            "(marginally different from inference)",
            file=sys.stderr,
        )
    return convert(args)


if __name__ == "__main__":
    raise SystemExit(main())
