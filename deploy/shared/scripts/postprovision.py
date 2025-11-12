#!/usr/bin/env python3
import sys
from pathlib import Path
import traceback
from utils import (
    load_azd_env_vars,
    parse_endpoints,
    get_openai_api_key,
    REPO_ENV_FILE,
    REPO_EXAMPLE_ENV_FILE,
)

YELLOW = "\033[33m"
RESET = "\033[0m"


def load_and_backup_env():
    """Backup existing .env (if any) and load base lines"""
    root_env = REPO_ENV_FILE
    if root_env.exists():
        backup_env = root_env.with_suffix(".bak")
        backup_env.write_bytes(root_env.read_bytes())
        print(f"Backed up existing .env to {backup_env}")
        base_file = root_env
    else:
        print(f"No existing .env found; using example at {REPO_EXAMPLE_ENV_FILE}")
        base_file = REPO_EXAMPLE_ENV_FILE
    lines = base_file.read_text().splitlines(True)
    return root_env, lines


def gather_env_values(env_vars):
    """Construct a dict of values to update in .env"""

    endpoints_str = env_vars.get("HLS_MODEL_ENDPOINTS")
    if not endpoints_str:
        raise RuntimeError("No endpoints found in AZD env; skipping .env update.")
    endpoints = parse_endpoints(endpoints_str)
    print(f"Parsed {len(endpoints)} endpoint(s) for update.")

    new_values = {}
    for ep in endpoints:
        name = ep.get("env_name")
        val = ep.get("id")
        if name and val:
            new_values[name] = val

    # Add standard AZD variables
    for key in (
        "AZURE_SUBSCRIPTION_ID",
        "AZURE_RESOURCE_GROUP",
        "AZUREML_WORKSPACE_NAME",
    ):
        v = env_vars.get(key)
        if v:
            new_values[key] = v

    # Add OpenAI variables if GPT deployment exists
    openai_endpoint = env_vars.get("AZURE_OPENAI_ENDPOINT")
    if openai_endpoint:
        new_values["AZURE_OPENAI_ENDPOINT"] = openai_endpoint
        deployment_name = env_vars.get("AZURE_OPENAI_DEPLOYMENT_NAME", "")
        if deployment_name:
            new_values["AZURE_OPENAI_DEPLOYMENT_NAME"] = deployment_name

        print(f"Found OpenAI endpoint: {openai_endpoint}")

        # Get AI Services name directly from deployment outputs
        ai_services_name = env_vars.get("AZURE_AI_SERVICES_NAME")
        rg_name = env_vars.get("AZURE_RESOURCE_GROUP", "")

        if ai_services_name and rg_name:
            print(f"Retrieving API key for AI Services: {ai_services_name}")

            try:
                api_key = get_openai_api_key(ai_services_name, rg_name)
                if api_key:
                    new_values["AZURE_OPENAI_API_KEY"] = api_key
                    print("Successfully retrieved OpenAI API key")
                else:
                    print("Warning: API key retrieval returned empty result")

            except Exception as e:
                print(f"Warning: Failed to retrieve OpenAI API key: {e}")
                print(
                    "You may need to retrieve this manually using: az cognitiveservices account keys list"
                )
        else:
            print(
                "Warning: AI Services name or resource group not found in deployment outputs"
            )

    return new_values


def merge_env_lines(lines, new_values):
    """Merge existing env lines with new values"""
    out = []
    seen = set()
    updates = []
    for line_no, line in enumerate(lines, 1):
        if "=" in line and not line.strip().startswith("#"):
            k, old_value = line.split("=", 1)
            key = k.strip()
            if key in new_values:
                seen.add(key)
                new_value = f'"{new_values[key]}"'
                if new_value == old_value:
                    continue
                new_line = f"{key}={new_value}\n"
                out.append(new_line)

                updates.append(f"- {line.strip()}")
                updates.append(f"+ {new_line}")
                continue
        out.append(line)
    out.append("\n\n")
    for key, val in new_values.items():
        if key not in seen:
            new_value = f'"{new_values[key]}"'
            new_line = f"{key}={new_value}\n"
            out.append(new_line)
            updates.append(f"+ {new_line}")
    return out, updates


def write_env(root_env, lines):
    """Write out the updated env file"""
    Path(root_env).write_text("".join(lines))
    print(f"Wrote updated .env to {root_env}\n")


def main():
    # Load current AZD environment
    env_vars = load_azd_env_vars()

    # Prepare and write .env update
    print("=== Updating repository .env file ===")
    root_env, base_lines = load_and_backup_env()
    new_values = gather_env_values(env_vars)

    print("Variables to update/add:")
    for k, v in new_values.items():
        print(f"  {k}={v}")
    merged, updates = merge_env_lines(base_lines, new_values)

    print("\nEnv file changes:")
    for line in updates:
        print(f"  {line}")
    print()
    write_env(root_env, merged)
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
