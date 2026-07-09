# Adapter Model Bundles

This directory holds the trained adapter model bundles used by `adapter.py`.

These bundles are small scikit-learn classifiers (an MLP head and an SVM head)
fit on MedImageInsight embeddings of chest X-rays. They predict the 15
ChestX-ray14 classes: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion,
Emphysema, Fibrosis, Hernia, Infiltration, Mass, No Finding, Nodule,
Pleural_Thickening, Pneumonia, and Pneumothorax. The included bundles were
re-serialized to load under this repository's pinned `numpy`/`scikit-learn`
versions.

| Bundle                       | Head | Size   |
|------------------------------|------|--------|
| `adapter_mlp/weights.joblib` | MLP  | ~3 MB  |
| `adapter_svm/weights.joblib` | SVM  | ~10 MB |

`adapter.py` uses the MLP head by default (`--head svm` selects the SVM head).

## Bundle contents

Each `weights.joblib` is a dict with:

| Key                | Description                                                  |
|--------------------|--------------------------------------------------------------|
| `model`            | Fitted sklearn `Pipeline` (`StandardScaler` + classifier) with `predict_proba` |
| `labels`           | Class names in column order of `predict_proba`               |
| `operating_points` | Per-label thresholds: `youden_j` (rule-in), `sens90` (rule-out) |

## Note on preprocessing

The bundles were trained on MedImageInsight embeddings of images prepared with a
specific pipeline (resize, JPEG quality, percentile normalization). At inference
the embedding is produced by `MedImageInsightClient`, so minor train/inference
preprocessing differences can affect calibration. The operating points were fit
during training and are applied as-is.

## Regenerating

To train fresh bundles instead of using the included ones:

- Adapter training: `azureml/medimageinsight/adapter-training.ipynb`
- Model comparison: `azureml/medimageinsight/model-comparison.ipynb`
