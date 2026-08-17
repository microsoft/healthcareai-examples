# Preparing your own data for MedImageParse3D fine-tuning

Fine-tuning runs the **2D** BiomedParse model over pre-sliced **2.5D RGB PNGs**
(`biomedparse_3D.forward_train` raises `NotImplementedError`). You therefore convert
volumes to slices offline, and the conversion has to match what the model sees at
inference or you introduce a train/serve skew.

[`prepare_finetune_data.py`](prepare_finetune_data.py) does that conversion and then
validates its own output.

## 1. Start from paired volumes

Match image and label by filename stem:

```
raw/
├── images/CT_AMOS_amos_0018.nii.gz
└── labels/CT_AMOS_amos_0018_seg.nii.gz     # --label-suffix _seg
```

Labels must be an integer class map: `0` = background, `1..N` = your targets.
NIfTI, MHA and NRRD are all accepted.

## 2. Convert

```bash
python prepare_finetune_data.py \
  --images-dir  raw/images \
  --labels-dir  raw/labels \
  --label-suffix _seg \
  --out         MY_DATASET \
  --classes     '{"1":"spleen","2":"right kidney","6":"liver","10":"pancreas"}' \
  --modality    CT \
  --site        abdomen \
  --ct-window   "40,400"
```

```
Found 1 case(s)
  CT_AMOS_amos_0018: axis=0 slices=63 kept=63 (+0 empty)

Wrote 63 slices (63 with foreground, 0 empty)
Validating MY_DATASET ...
Valid: 63 slices, class ids [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
```

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--ct-window "40,400"` | CT level/width; omit for 0.5–99.5 percentile scaling |
| `--negative-ratio 0.15` | empty slices kept as negatives, as a fraction of foreground slices |
| `--instance-label` | masks hold instance ids rather than class ids |
| `--validate-only DIR` | lint a dataset prepared some other way |

## 3. Result

```
MY_DATASET/
├── train/        512×512 RGB PNG   (uint8)
├── train_mask/   512×512 grayscale (uint8, pixel value == class id)
└── train.json
```

```json
{
  "class_prompts": {"6": ["liver", "liver in abdomen CT", "CT imaging of liver in abdomen", "..."]},
  "annotations": [
    {"mask_file": "CT_AMOS_amos_0018_000.png", "file_name": "CT_AMOS_amos_0018_000.png",
     "split": "train", "class_ids": [15], "instance_label": 0}
  ]
}
```

No `test/` split is needed — the trainer carves validation out of train
(`split_train_validate: True`, `validate_split_ratio: 0.1`).

## 4. Upload and train

The dataset root is the mount root, so register the folder as-is and the existing
pipeline in `medimageparse_3d_finetuning.ipynb` needs no other change:

```python
training_data = Data(path="MY_DATASET", type=AssetTypes.URI_FOLDER,
                     name="medimageparse3d-training_data")
training_data = ml_client.data.create_or_update(training_data)
```

## What the conversion does, and why

Each step mirrors inference (`utils.process_input` / `forward_eval`):

- **Slice axis** via the same shape heuristic as `utils.get_axis`.
- **Centre zero-pad to square, resize to 512** (bicubic image, nearest mask).
- **Intensities to 0–255** — the model normalises with a single scalar mean/std
  (`gray_scale=True`), so the absolute range matters more than per-slice contrast.
  Statistics are taken over the whole volume so neighbouring slices stay consistent.
- **2.5D packing**: `ch0 = slice d`, `ch1 = slice d-1`, `ch2 = slice d+1`, with the
  first and last slice borrowing their neighbour's neighbour. Verified byte-identical
  against the shipped `CVPR-MR-crossmoda-sample`.

## Pitfalls

- **Binary `{0, 255}` masks are silently discarded.** The dataset zeroes *both* the
  image and the mask for them and prints one line; the job still reports success and
  learns nothing. Encode pixels as class ids. `--validate-only` fails loudly on this.
- **Per-annotation `class_ids` is never read.** Sampling is driven by the keys of the
  top-level `class_prompts`, so every slice samples `num_prompts=4` ids from *all*
  classes. Classes absent from a slice yield empty masks on purpose — they train the
  `object_existence` head. To sample only classes present in a slice, drop the
  top-level `class_prompts` and give each annotation its own.
- **Prompt wording matters.** One paraphrase is drawn at random per sample; include
  the phrasing you intend to use at inference.
- **The 10% validation split is random over slices**, so adjacent slices from one
  patient land on both sides. For a trustworthy metric, hold out whole patients.
- **`trainer.devices` defaults to 1.** Multi-GPU needs both `devices:` in
  `parameters.yaml` and `process_count_per_instance` on the component.
