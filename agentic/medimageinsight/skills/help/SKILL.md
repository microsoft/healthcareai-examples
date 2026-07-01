---
name: help
description: Entry point for the MedImageInsight plugin, covering the getting-started walkthrough, how the skills work, what MI2 is, and troubleshooting.
argument-hint: ask a question, or leave blank for the menu
---

# help

**Locating plugin files.** This skill lives at `skills/help/`. Resolve all asset paths as absolute paths from this skill's own directory. Reference docs are in `references/`, a sibling of this file. MCP server source is at `../../mcp-server/`. Do NOT rely on the `CLAUDE_PLUGIN_ROOT` environment variable; it is not reliable in all environments.

**Tool-ID note.** Some agent hosts shorten the server name to `medimageinsig` in tool IDs. The four registered tools are:

- `mcp_medimageinsig_zeroshot_classify`
- `mcp_medimageinsig_adapter_classify`
- `mcp_medimageinsig_setup`
- `mcp_medimageinsig_zeroshot_label_examples`

## Behavior

### Blank invoke, or "what does this do"

Print the following block, then wait for the user to pick a number:

---

This plugin gives an agent grounded chest X-ray reading. It wires MedImageInsight (a domain vision model) to the Radiology Assistant agent through MCP tools. The agent calls MI2 to classify findings from an image, then drafts a structured radiologist report. Reports are research and educational examples only, not for clinical use.

What would you like to do?

1. Walk me through a first report
2. How the skills work
3. More about MedImageInsight
4. Help and testing

---

### Freeform invoke

Skip the menu. Classify the request and load the matching reference file as an absolute path from this skill's own directory, then follow its instructions.

| If the user mentions... | Load |
|---|---|
| error, not working, broken, slow, mcp, tools don't appear | `references/troubleshooting.md` |
| how do the skills work, skill architecture, why skills | `references/skills.md` |
| about, what is mi2, what is medimageinsight, model, adapter | `references/about.md` |
| walk me through, walkthrough, getting started, first report, tutorial | `references/walkthrough-agent.md` |

Menu items map to the same files:

1. `references/walkthrough-agent.md`
2. `references/skills.md`
3. `references/about.md`
4. `references/troubleshooting.md`
