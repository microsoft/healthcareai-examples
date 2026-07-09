# About MedImageInsight

## Why a domain vision model

A general language model given a chest X-ray file path has no image access. It fills in findings from training data, producing plausible-sounding reports that may not reflect what is actually in the image. MedImageInsight (MI2) is a domain-specific vision model trained on medical imaging data. Wiring it through an MCP tool gives the agent a grounded signal: the agent asks MI2 "what is in this image?" and writes the report from those answers instead of guessing.

The architecture is intentionally narrow. You add a tool, the agent gains a capability, nothing else changes. The same agent file works with zero-shot, adapter, or both. That is the core promise of MCP: add tools, don't rewrite the harness.

---

## Zero-shot vs adapter

| | Zero-shot (`ground-zeroshot`) | Adapter (`ground-adapter`) |
|---|---|---|
| **Setup** | None | Train on labeled CXR data |
| **Labels** | Any text labels at call time | Fixed set from training |
| **Output** | Raw probabilities, bucketed by the skill | Pre-bucketed: positive / possible / negative |
| **Best when** | No labeled data; quick first pass | Labeled data available; calibration matters |

**Zero-shot** calls `mcp_medimageinsig_zeroshot_classify`. You pass any text labels and get back a probability per label. The `ground-zeroshot` skill applies fixed cutoffs (>=0.20 positive, 0.08-0.20 possible) to produce the finding list.

**Adapter** calls `mcp_medimageinsig_adapter_classify`. A small classifier head was trained on MI2 embeddings with calibrated per-class thresholds. The tool returns buckets directly. One-line provenance: pretrained MI2 model, frozen embeddings, small classifier head, calibrated thresholds from training.

---

## Default label set (ChestX-ray14)

When zero-shot runs without custom labels, the `ground-zeroshot` skill uses these 15 findings from the ChestX-ray14 dataset:

Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, No Finding, Nodule, Pleural_Thickening, Pneumonia, Pneumothorax.

You can append additional text labels at call time via `extra_labels`. Zero-shot accepts arbitrary text; the adapter is fixed to the label set it was trained on.

---

## What good output looks like

A correctly grounded report follows this exact format (four all-caps sections, plain text, no markdown):

```
INDICATION: Not provided.
TECHNIQUE: Frontal chest radiograph.
FINDINGS: Patchy opacity in the right lower lobe, compatible with consolidation. The remaining lungs are clear. No pleural effusion or pneumothorax. Cardiomediastinal silhouette is within normal limits. No acute osseous abnormality. No support devices.
IMPRESSION:
1. Right lower lobe consolidation, compatible with pneumonia.
2. No pleural effusion or pneumothorax.
```

The expected agent trace for either grounding path:

1. One `mcp_medimageinsig_zeroshot_classify` or `mcp_medimageinsig_adapter_classify` call.
2. `write-report` skill invoked to format findings into the four-section report.
3. Clean `<report>` block with no probability scores, model names, or classification details.

Grounding details (method, scores, confidence) appear in the `<info>` block above the report, not inside `<report>`.

**Reports produced here are research and educational examples only, not for clinical use.**

---

For a guided first run, see [walkthrough.md](../../../walkthrough.md). For how the skills divide work, see [skills.md](skills.md).
