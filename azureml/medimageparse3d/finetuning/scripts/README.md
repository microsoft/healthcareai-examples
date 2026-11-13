# Command-Line Scripts

Run commands from the parent `finetuning/` directory, using the environment in
the [setup instructions](../README.md#setup). Every module supports `--help` and
can be imported without starting a job or deployment.

| Module | Purpose |
| --- | --- |
| [prepare_data](prepare_data.py) | Write training/validation slices or scoring volumes. |
| [train](train.py) | Submit the train, score, and evaluate pipeline. |
| [aml_job_status](aml_job_status.py) | Check or wait for the parent pipeline. |
| [get_metrics](get_metrics.py) | Download the completed evaluation CSV. |
| [deploy_model](deploy_model.py) | Reuse a deployment or register and deploy the model. |
| [test_endpoint](test_endpoint.py) | Invoke a deployment and save a prediction NIfTI. |

## Workspace

Azure commands use your `az login` session. Place a local `config.json` in
`finetuning/`, or pass `--workspace-config /path/to/config.json`:

```json
{
  "subscription_id": "<subscription-id>",
  "resource_group": "<resource-group>",
  "workspace_name": "<workspace-name>"
}
```

This file is ignored by Git. Do not include credentials in it. Metrics downloads
also require blob read access to the workspace's default datastore.

## Prepare Data

Use the notebook's case mappings and class configuration as JSON inputs. The
converter does not select patients or create a train/test split. Supply disjoint
case lists yourself; for BHSD, follow the notebook's official split and exclusions.

Example `inputs/train-cases.json`:

```json
{
  "case-001": {
    "image": "/path/to/images/case-001.nii.gz",
    "gt": "/path/to/masks/case-001.nii.gz"
  }
}
```

Paths may be absolute or relative to the cases JSON file. Case IDs become output
filenames, so use filename stems without directory separators. Create separate
case files for validation and test volumes.

Example `inputs/classes.json` for a single class:

```json
{
  "class_names": {"1": "epidural hemorrhage"},
  "class_prompts": {"1": ["CT head showing epidural hemorrhage"]}
}
```

For the BHSD recipe, include all five classes and the seven phrases per class
shown in the notebook. JSON keys are strings; mask values remain integer class IDs.

```bash
python -m scripts.prepare_data slices \
  --cases inputs/train-cases.json --classes inputs/classes.json \
  --output-dir prepared/bhsd/training \
  --window-width 200 --window-level 40 \
  --target-repeat '{"1": 5, "5": 5}' --negative-sample-ratio 0.15 --seed 0

python -m scripts.prepare_data volumes \
  --cases inputs/test-cases.json --classes inputs/classes.json \
  --output-dir prepared/bhsd/evaluation --window-width 200 --window-level 40
```

For internal validation, call `slices` again with separate held-out cases and the
same training output root:

```bash
python -m scripts.prepare_data slices \
  --cases inputs/val-cases.json --classes inputs/classes.json \
  --output-dir prepared/bhsd/training --split val \
  --window-width 200 --window-level 40 \
  --target-repeat '{}' --negative-sample-ratio 0.15 --seed 0
```

`train` is the default split. `val` writes `val/`, `val_mask/`, and `val.json`
without changing training files. Repetition defaults to `{}` in the CLI; pass
the mapping explicitly to reproduce the notebook's five-copy training recipe.
Validation is not enabled automatically: update [parameters.yaml](../parameters.yaml)
as described in the notebook. Preparation refuses to overwrite existing split
directories; use a new output directory when changing the recipe.

## Train and Monitor

```bash
python -m scripts.train \
  --training-data prepared/bhsd/training \
  --evaluation-data prepared/bhsd/evaluation \
  --compute-name '<gpu-compute-name>' \
  --experiment-name mip3d-bhsd-finetuning \
  --job-name '<new-parent-job-name>'
```

Data arguments also accept Azure ML data-asset or datastore URIs. The script
resolves the base model's `latest` label and records its serving environment.
It uses component `0.0.2`, the existing scoring/evaluation components, and the
same output layout as the notebook. `--parameters` selects another configuration
file; keep its basename `parameters.yaml`. Add `--force-rerun` to disable pipeline
step reuse, or `--wait` to stream the submitted parent through evaluation.

Use the printed **parent** job name for status and results, not its training child:

```bash
python -m scripts.aml_job_status '<parent-job-name>' --status
python -m scripts.aml_job_status '<parent-job-name>' --timeout-minutes 360 --poll-seconds 30
python -m scripts.aml_job_status '<parent-job-name>' --wait-for Running
```

Without `--status`, the watcher waits for completion or the requested state.
It prints state changes and exits with `0` on success, `1` on terminal failure,
or `2` on timeout. Status-only checks exit `0` after printing the current state.
Interrupting the watcher does not cancel the remote job.

## Download Metrics

```bash
python -m scripts.get_metrics --job-name '<completed-parent-job-name>'
```

The CSV is written to `results/<job-name>/metrics.csv`; override this with
`--output-dir`. The helper uses this example's default-datastore output layout
and Azure CLI authentication without browser or device-code prompts. At the
time of writing (September 2026), it works around an Azure ML SDK named-output
download bug.

## Deploy and Invoke

```bash
python -m scripts.deploy_model \
  --job-name '<completed-parent-job-name>' --model-name mip3d-bhsd \
  --endpoint-name '<unique-endpoint-name>' --deployment-name ich \
  --instance-type Standard_NC40ads_H100_v5

python -m scripts.test_endpoint \
  --endpoint-name '<unique-endpoint-name>' --deployment-name ich \
  --evaluation-data prepared/bhsd/evaluation --sample-index 0 \
  --output results/prediction.nii.gz
```

A successful existing deployment is reused without changing its model or traffic.
Check the displayed model; reuse does not establish that it came from the supplied
job. A missing deployment registers the job's exported model and provisions one
GPU instance. **A new deployment receives 100% of endpoint traffic**, including on
an existing endpoint. Use a separate endpoint to avoid changing shared routing.

Inference targets the named deployment and saves its decoded prediction. It does
not register, deploy, or delete resources. Input volumes must already have the
same windowing used for training. Use the notebook for CT/label/prediction plots.
Delete unneeded deployments through the notebook's optional cleanup or Azure ML
Studio; zero traffic does not stop compute charges.

## Python Imports

```python
from scripts.prepare_data import prepare_slices, prepare_volumes
from scripts.train import create_training_job
from scripts.get_metrics import download_metrics
from scripts.deploy_model import deploy_model
from scripts.test_endpoint import predict_volume
from scripts.aml_job_status import check_job
```

The notebook calls these same functions. `create_training_job` only builds the
pipeline; submission remains explicit through `ml_client.jobs.create_or_update`.