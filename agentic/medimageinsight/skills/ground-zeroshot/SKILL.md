---
name: ground-zeroshot
description: Ground CXR findings using MI2 zero-shot classification. Returns per-finding probabilities bucketed into positive/possible for the agent to use when writing the report.
argument-hint: |
  - image_path: required file path or http(s) URL to the CXR image (PNG or JPEG)
  - extra_labels: optional list of additional label strings to score alongside the default set
---

# ground-zeroshot

Calls the MI2 zero-shot classifier and returns bucketed per-finding probabilities. Does not write a report — the agent produces the report using the `write-report` skill after grounding.

## Budget

- Tool calls: 1 — exactly one `mcp_medimageinsig_zeroshot_classify` call per image.
- Do not retry on success.

## How to use

1. **Resolve the image.** If `image_path` is an http(s) URL, pass it through to the tool **unchanged** — do NOT download it, `curl` it, or turn it into a local file; the MCP server fetches URLs itself. Otherwise it is a local path: the MCP tool runs in a different working directory, so if it is not absolute, resolve it via `realpath`; if it can't be resolved, ask the user.

2. **Assemble the label list.** Read the canonical label list from below and use those strings **verbatim**. Append any caller-provided `extra_labels`.

3. **Call the classifier.** Call `mcp_medimageinsig_zeroshot_classify` with `file_path` = absolute path and `labels` = assembled list. One call per image.

4. **Bucket the scores.** Use these cutoffs:
   - **≥ 0.20** → positive
   - **0.08 – 0.20** → possible
   - **< 0.08** → absent (omit)

5. **Return the findings.** Return the positive and possible labels with their raw scores. Do not write a report. The agent decides what to do next.


## NIH ChestX-ray14 labels (MI2 format)

Pass these strings as the `labels` argument to `zeroshot_classify`.
Format is `"<modality> <body_part> <view> <condition>"` — MI2 is sensitive to
this structure, do not reword.

```
x-ray chest anteroposterior Atelectasis
x-ray chest anteroposterior Cardiomegaly
x-ray chest anteroposterior Consolidation
x-ray chest anteroposterior Edema
x-ray chest anteroposterior Effusion
x-ray chest anteroposterior Emphysema
x-ray chest anteroposterior Fibrosis
x-ray chest anteroposterior Hernia
x-ray chest anteroposterior Infiltration
x-ray chest anteroposterior Mass
x-ray chest anteroposterior Nodule
x-ray chest anteroposterior Pleural_Thickening
x-ray chest anteroposterior Pneumonia
x-ray chest anteroposterior Pneumothorax
```
