---
name: write-report
description: Read a chest X-ray end-to-end: ground findings with MedImageInsight, then write a structured radiologist report (Indication, Technique, Findings, Impression). Invoke for any "write/draft a CXR report" request.
argument-hint: |
  - image: required — file path or http(s) URL to the CXR image (PNG or JPEG)
  - method: optional — grounding method: zeroshot (default), adapter, or none
  - indication: optional — clinical context / reason for exam
---

# write-report

Runs the full CXR report flow: resolve the image, ground findings with MedImageInsight, then write a structured radiologist report. Invoke as: `write-report <zeroshot|adapter|none> <image> [indication]`.

## Budget

Grounding is at most one tool call per image, made inside the grounding skill. Report-writing itself calls no tools. Do not retry on success.

## Workflow

Print this checklist and work through it in order, checking off each step as you go.

1. [ ] **Announce the checklist** — print it so the user sees the plan before you begin.
2. [ ] **Resolve the image** — if `image` is an http(s) URL, use it as-is (do not check the filesystem). Otherwise turn it into an absolute path and confirm the file exists using your tools; if it does not resolve to a real file, STOP and ask the user for a valid path or URL. Note `indication` and any priors.
3. [ ] **Pick the grounding method** — use `method` if given, otherwise default to `zeroshot`. `none` means an unaided read with no grounding.
4. [ ] **Ground the findings** — open the grounding skill for the chosen method and follow it exactly:
   - `zeroshot` → [`ground-zeroshot`](../ground-zeroshot/SKILL.md)
   - `adapter` → [`ground-adapter`](../ground-adapter/SKILL.md)
   - `none` → skip; read the image unaided.

   Follow the linked skill's own steps to get the per-finding calls — do not call the MI2 tools directly and do not invent the label set. If grounding errors, STOP (see **On error**).
5. [ ] **Write the report** — using the findings, write it in the exact format in **Report format** below.
6. [ ] **Emit the output envelope** — see **Output**.

## Report format

Use this section structure, in this order, with all-caps headers:

```
INDICATION:
TECHNIQUE:
FINDINGS:
IMPRESSION:
```

- **INDICATION** — one short sentence on clinical context. If none was provided, write: `Not provided.`
- **TECHNIQUE** — one sentence describing the study (view, technique). Default: `Frontal chest radiograph.`
- **FINDINGS** — impersonal declarative sentences. Mention all standard structures (lungs, pleura, heart, mediastinum, bones, any devices) even when normal.
- **IMPRESSION** — **numbered**, prioritized by clinical significance. One concise statement per number. Hedging language permitted.
- Plain text only — no markdown. The all-caps section labels are the only headers.

Borrow phrasings from [`references/phrase-bank.md`](references/phrase-bank.md).

### Style rules

- **FINDINGS** states observations; **IMPRESSION** states interpretations.
- Do not use markdown formatting, plain text only. No bold, italics, or lists.
- Match hedge strength to evidence:
  - High evidence or high scores: state directly. `Right lower lobe pneumonia.`
  - Moderate: use `possible` or `differential includes`.
  - Low but non-zero: use `cannot exclude`.
  - Absent: omit, or state the negative if clinically relevant.
- State negatives explicitly. Group them: `No pleural effusion or pneumothorax.`
- No padding, filler, or clinical disclaimers.
- Present tense.

### Critical rule

**The report must read as the radiologist's direct interpretation of the image. Do not mention the tools, models, classifiers, scores, thresholds, or any system internals used to produce the findings.**

## Output

The skill emits exactly this envelope — `<info>` holds the grounding details (method, evidence, confidence); `<report>` holds the clean radiologist report. Match this structure exactly:

```
<output>
<info>
Grounding: zeroshot
Evidence: Consolidation 0.62 (positive), Pneumonia 0.48 (positive), Effusion 0.06 (absent)
Confidence: medium
Note: research/educational example, not for clinical use.
</info>
<report>
INDICATION: Cough and fever.
TECHNIQUE: Frontal AP chest radiograph.
FINDINGS: Patchy opacity in the right lower lobe, compatible with consolidation. The remaining lungs are clear. No pleural effusion or pneumothorax. Cardiomediastinal silhouette is within normal limits for technique. No acute osseous abnormality. No support devices.
IMPRESSION:
1. Right lower lobe consolidation, compatible with pneumonia.
2. No pleural effusion or pneumothorax.
</report>
</output>
```

- `Grounding` — `none`, `zeroshot`, or `adapter (head)`.
- `Evidence` — the per-finding calls/scores from grounding, or `unaided read` for `none`.
- `Confidence` — `low`, `medium`, or `high`, derived from the grounding evidence and image quality. Use `low` when image quality limits the read, and note that in FINDINGS.
- `<report>` — the clean report only: no `<info>` content, no method/model/score talk.

## On error

If image resolution or grounding fails, STOP. In plain prose, say what failed and the likely fix (for example: run `setup`, or check that the image path is correct and reachable). Emit NO `<output>`, `<info>`, or `<report>` block — not even an empty one. Never retry blindly or write an ungrounded report.
