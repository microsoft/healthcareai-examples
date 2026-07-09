# MedImageInsight Agentic Example

A self-contained Copilot/Claude plugin that wraps MedImageInsight (MI2) for chest X-ray reading.
It exposes MI2 classification as MCP tools, includes a Radiology Assistant agent that
orchestrates the full read-and-report workflow, provides skills for grounding findings and
writing structured reports, and bundles two NIH ChestX-ray14 sample images for smoke testing.

For educational and example use only. Not for clinical use.

---

## Install

Register the repository as a marketplace source, then install the plugin:

```bash
copilot plugin marketplace add https://github.com/microsoft/healthcareai-examples
copilot plugin marketplace browse healthcareai-examples
copilot plugin install medimageinsight@healthcareai-examples
```

### Local (development)

Install directly from a clone of this repo:

```bash
cd healthcareai-examples/agentic
copilot plugin install ./medimageinsight
```

> [!IMPORTANT]
> Run the install from the `agentic` directory (as shown above), or the relative path won't resolve.
> Local install also behaves differently in VS Code vs the Copilot CLI; after installing or changing the plugin, you may need to reinstall and reload the VS Code window before changes take effect.
> If the plugin misbehaves, try the `help` skill.

## Configure

The MCP server reads one required environment variable:

```
MI2_MODEL_ENDPOINT=<AzureML online-endpoint resource ID>
```

This must be the full AzureML online-endpoint resource ID string, not a plain URL. The server
fetches the API key server-side; do not pass the key through chat.

Put this in a `.env` file. The server does not auto-discover it — pass the file explicitly with
`setup(env_file=/path/to/.env)`, or have your host inject it via `env` / `envFile` in `.mcp.json`.
If you have already run the [healthcareai-examples](https://github.com/microsoft/healthcareai-examples)
notebooks, reuse that repo's `.env`. To deploy the model fresh, follow the
[deployment guide](https://github.com/microsoft/healthcareai-examples/blob/main/docs/deployment-guide.md),
which writes `MI2_MODEL_ENDPOINT` into your `.env`.

## Try it

The guided walkthrough is the fastest start. In a chat, invoke the plugin's help skill and pick the first option:

```
/medimageinsight help
```

It walks you through verifying the tools, running setup, and generating your first report. You can also describe a problem to jump straight to troubleshooting, for example `/medimageinsight help the mcp server is not working`.

> [!TIP]
> To read it first, see the [walkthrough](walkthrough.md).

Once set up, open the **Radiology Assistant** agent and ask it to read an image. The agent accepts a local file path or an http(s) URL; it cannot process images attached directly to chat.

```
Please write a radiology report for https://media.githubusercontent.com/media/microsoft/healthcareai-examples-data/main/medimageinsight/plugin-samples/00026132_011.png
```

To use your own image, give it an absolute path.

## How it fits together

**MCP server** (`mcp-server/`) exposes four tools:

| Tool | What it does |
|------|--------------|
| `setup` | Configure MI2 from an env file or endpoint resource ID. |
| `zeroshot_label_examples` | Return the canonical CXR label set (local; no endpoint needed). |
| `zeroshot_classify` | Run MI2 zero-shot classification; returns per-label probabilities. |
| `adapter_classify` | Run the trained MI2 adapter; returns per-finding positive/possible/negative calls. |

**Agent** (`agents/radiology-assistant.agent.md`, GPT-5 mini) is a thin radiology persona: it
invokes `write-report` and enforces the guardrails (images by path or URL, stop on errors, no
system internals in the report). The orchestration itself lives in `write-report`.

**Skills** (`skills/`):

| Skill | Role |
|-------|------|
| `ground-zeroshot` | Calls `zeroshot_classify`; returns findings bucketed by score (positive / possible). |
| `ground-adapter` | Calls `adapter_classify`; returns positive/possible findings. |
| `write-report` | Orchestrator: resolves the image, picks the grounding method, calls the matching grounding skill, writes the structured report (INDICATION, TECHNIQUE, FINDINGS, IMPRESSION), and emits the output envelope. |
| `help` | Entry point for the plugin: a guided first-report walkthrough, how the skills work, what MI2 is, and troubleshooting. Run `/medimageinsight help` for the menu, or describe a problem for direct help. |

The report produced by `write-report` reads as the radiologist's direct interpretation of the
image. It does not mention models, classifiers, scores, or system internals.

## Layout

```
.claude-plugin/plugin.json   plugin manifest
.mcp.json                    MCP server config
agents/
  radiology-assistant.agent.md
skills/
  ground-zeroshot/
  ground-adapter/
  write-report/
  help/
mcp-server/                  Python MCP server (uv)
```

## Smoke test (local)

From the plugin root:

```bash
cd mcp-server
uv run python smoketest.py --quiet --env-file /path/to/.env
```

## Troubleshooting

### MCP server exits with `No such file or directory (os error 2)`

The `help` skill automates the diagnosis and fix below. Run `/medimageinsight help the mcp server is not working`, or follow
the steps manually.

In some VS Code Remote-SSH environments the MCP server starts from the wrong directory. If the
MCP log shows:

```text
error: No such file or directory (os error 2)
```

edit the INSTALLED plugin's `.mcp.json` (not the repo source). It is usually under:

```text
~/.copilot/installed-plugins/healthcareai-examples/medimageinsight/.mcp.json
```

If you installed locally, it lands under `_direct/medimageinsight/` instead. The `help`
skill locates the right file and applies the fix for you.

Add a `cwd` field pointing to the installed plugin root:

```json
{
  "mcpServers": {
    "medimageinsight": {
      "command": "uv",
      "args": ["run", "--directory", "./mcp-server", "python", "server.py"],
      "cwd": "/home/<user>/.copilot/installed-plugins/healthcareai-examples/medimageinsight"
    }
  }
}
```

Replace `/home/<user>/...` with your actual home directory path. Restart the MCP server after
saving.

## Disclaimer

This example is for research and development exploration only. It is not designed or intended
for deployment in clinical settings or for use in the diagnosis or treatment of any health or
medical condition.
