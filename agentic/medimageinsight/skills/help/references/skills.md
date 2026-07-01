# How the skills work

## The split

Three skills handle CXR reporting, and each does exactly one thing:

- **`ground-zeroshot`** calls `mcp_medimageinsig_zeroshot_classify`, buckets the raw probabilities, and returns a finding list. It does not write a report.
- **`ground-adapter`** calls `mcp_medimageinsig_adapter_classify`, reads the pre-bucketed positive/possible/negative results, and returns a finding list. It does not write a report.
- **`write-report`** is the orchestrator: it picks the grounding method, invokes the matching grounding skill for findings, writes the structured four-section radiologist report, and emits the output envelope. It does not call the MI2 tools directly — grounding always goes through a grounding skill.

The Radiology Assistant agent invokes `write-report`, which selects the grounding method, calls the matching grounding skill for findings, writes the report, and wraps it in the output envelope. Orchestration lives in `write-report`; the agent is a thin radiology persona over it.

---

## MCP framing

Each MI2 classification tool is a Python function registered with the MCP server. The agent host loads the server at startup, presents its tools to the agent, and routes calls through the server process. The agent treats the tools as functions it can invoke by name.

The agent file and the skills stay unchanged when tools are added or swapped. That is the core promise: add tools, don't rewrite the harness.

---

## Zero-shot mechanics

`ground-zeroshot` uses MI2's embedding space and cosine similarity to text labels. You pass any labels as strings. The model returns a probability per label. The skill applies fixed cutoffs:

- >=0.20: positive
- 0.08-0.20: possible
- <0.08: absent (omitted from findings)

---

## Adapter mechanics

`ground-adapter` runs the image through MI2 to get an embedding, then passes that embedding to a small classifier head trained on labeled CXR data. The head was fit with per-class operating points (Youden-J for rule-in, sensitivity-90 for rule-out). The tool returns `positive`, `possible`, or `negative` per label. The skill reads these directly with no further thresholding.

One-line provenance: pretrained MI2 model, frozen embeddings, small classifier head, calibrated thresholds from training.

---

## Why the skills don't call each other

Keeping grounding and report-writing separate makes each skill testable and replaceable independently. `write-report` can receive findings from any source, not just MI2. The agent decides confidence and context before deciding how to phrase the report.

---

For background on MI2 and the two grounding approaches, see [about.md](about.md). For setup and troubleshooting, see [troubleshooting.md](troubleshooting.md).
