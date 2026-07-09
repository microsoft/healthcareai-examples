---
name: ground-adapter
description: Ground CXR findings using the trained MI2 adapter. Returns per-finding positive/possible/negative calls for the agent to use when writing the report.
argument-hint: |
  - image_path: required file path or http(s) URL to the CXR image (PNG or JPEG)
  - head: optional adapter head — "mlp" (default) or "svm"
---

# ground-adapter

Calls the trained MI2 adapter and returns per-finding positive/possible/negative calls. Does not write a report — the agent produces the report using the `write-report` skill after grounding.

## Budget

- Tool calls: 1 — exactly one `mcp_medimageinsig_adapter_classify` call per image.
- Do not retry on success.

## How to use

1. **Resolve the image.** If `image_path` is an http(s) URL, pass it through to the tool **unchanged** — do NOT download it, `curl` it, or turn it into a local file; the MCP server fetches URLs itself. Otherwise it is a local path: the MCP tool runs in a different working directory, so if it is not absolute, resolve it via `realpath`; if it can't be resolved, ask the user.

2. **Call the adapter.** Call `mcp_medimageinsig_adapter_classify` with `file_path` = absolute path and `head` = the caller's value if provided (otherwise omit; defaults to `mlp`). One call per image.

3. **Read the results.** The adapter returns `{label: "positive" | "possible" | "negative"}`, already bucketed using per-class operating points fit during training (Youden-J for rule-in, sens90 for rule-out). No further thresholding.

4. **Return the findings.** Return the positive and possible labels. Omit negatives. Do not write a report. The agent decides what to do next.
