# MedImageParse3D Fine-Tuning

Fine-tune MedImageParse3D on paired CT volumes and segmentation masks, evaluate
the exported model, and call a managed online deployment.

- [Notebook](mip3d-finetuning.ipynb): BHSD walkthrough with case selection,
  prompts, training settings, metrics, and slice comparisons.
- [Scripts](scripts/README.md): the same preparation, pipeline, download,
  deployment, and inference functions with command-line entry points.
- [Training parameters](parameters.yaml): batch size, optimizer, epochs, and
  validation settings passed to fine-tuning component `0.0.2`.

## Setup

Use a fresh Python 3.11 environment. From the repository root, install the
package with its `examples` extra:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e './package[examples]'
az login
cd azureml/medimageparse3d/finetuning
```

The [package](../../../package/pyproject.toml) supports Python 3.10 and 3.11 and
provides the local dependencies, including OpenCV, plotting, and notebook support.
Do not also install `opencv-python-headless`: the package uses `opencv-python`,
and both provide the same `cv2` module.

Select that environment as the notebook kernel, or follow the
[CLI instructions](scripts/README.md). Training needs an Azure ML workspace,
GPU compute, and a compute identity with access to the data and model registries.
The supplied recipe uses one GPU and BF16 mixed precision.

Download [BHSD](https://huggingface.co/datasets/Wendy-Fly/BHSD) and follow the
notebook's directory layout and exclusions. Training preparation supports NIfTI,
MHA, and NRRD; scoring preparation expects NIfTI. Image/mask pairs must already
be aligned. Use the same window and class configuration for training and scoring.

## Reuse a Completed Run

Set `completed_job_name` in the notebook to a completed parent pipeline and
`prepared_dir` to its matching prepared data. Preparation and submission are
skipped. A successful named deployment is reused unchanged; verify its displayed
model before invoking it. CLI commands accept the parent job name directly.

New training jobs and deployments incur charges. Reusing an endpoint does not
stop compute billing. Cleanup is optional and deletes the selected endpoint and
all its deployments. Local config, prepared data, and results are Git-ignored.