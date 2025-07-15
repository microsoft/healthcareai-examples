#!/usr/bin/env python3
import sys
import argparse
from utils import (
    get_model_filter,
    ensure_azd_env,
    load_models,
    load_azd_env_vars,
    get_ml_workspace,
)
import traceback

# ANSI colors
YELLOW = "\033[93m"
RED = "\033[91m"
END = "\033[0m"


def main(yes: bool = True, validate_existing: bool = False):
    ensure_azd_env()

    # Gather environment info first
    env = load_azd_env_vars()
    env_name = env["AZURE_ENV_NAME"]
    subscription = env["AZURE_SUBSCRIPTION_ID"]

    rg_name = env.get("AZURE_RESOURCE_GROUP", f"rg-{env_name} (assumed)")
    ws_name = env.get("AZUREML_WORKSPACE_NAME", f"mlw-{env_name} (assumed)")

    # Validate existing workspace if requested
    if validate_existing:
        print(
            f"Validating existing workspace '{ws_name}' in resource group '{rg_name}'..."
        )
        try:
            ws_obj = get_ml_workspace(ws_name, rg_name, subscription)
            print(f"✓ Workspace found: {ws_obj['name']} in {ws_obj['location']}")

            # Validate location matches
            ws_location = ws_obj["location"]
            current_loc = env.get("AZURE_LOCATION")
            if current_loc and current_loc.lower() != ws_location.lower():
                print(
                    f"{RED}ERROR: AZURE_LOCATION ({current_loc}) does not match workspace location ({ws_location}){END}"
                )
                return 1

        except RuntimeError as e:
            print(f"{RED}ERROR: {e}{END}")
            print(
                f"{RED}Please ensure the workspace exists or check your configuration.{END}"
            )
            return 1

    models = load_models()
    model_filter = get_model_filter()
    models_to_deploy = []
    for model in models:
        name = model.get("name", "<unknown>")
        if not model_filter or name in model_filter:
            deployment = model.get("deployment", {})
            instance_type = deployment.get("instanceType", "<not set>")
            instance_count = deployment.get("instanceCount", "<not set>")
            models_to_deploy.append((name, instance_type, instance_count))

    # Check for GPT deployment configuration
    gpt_model = env.get("gptModel", "").strip()
    gpt_capacity = env.get("gptModelCapacity", "50")
    gpt_location = env.get("gptDeploymentLocation", env.get("AZURE_LOCATION", ""))

    print(f"AZD Environement: {env_name}")
    print(f"\nThe following models will be deployed to Azure ML workspace: {ws_name}")
    print(f"Resource group: {rg_name}")
    print(f"Subscription: {subscription}")
    print(f"These models will incur Azure charges.\n")

    # Display healthcare AI models
    for name, instance_type, instance_count in models_to_deploy:
        print(f"- {name}: {instance_type} x {instance_count}")

    # Display GPT deployment if configured
    if gpt_model:
        model_name, model_version = (
            gpt_model.split(";") if ";" in gpt_model else (gpt_model, "latest")
        )
        print(f"\nGPT model deployment:")
        print(f"- {model_name} (version: {model_version})")
        print(f"  Capacity: {gpt_capacity}K tokens per minute")
        print(f"  Location: {gpt_location}")

    # Continue to confirmation prompt
    if not yes:
        print("\nContinue with deployment? [y/N]: ", end="")
        choice = input().strip().lower()
        if choice not in ("y", "yes"):
            print(f"{RED}Aborting deployment.{END}")
            return 1
    else:
        print(
            f"{YELLOW}Skipping confirmation (--yes). Proceeding with deployment...{END}"
        )
    print(f"{YELLOW}Proceeding with deployment...{END}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Assume yes for all confirmation prompts",
    )
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="Validate that the workspace exists in the specified resource group",
    )
    args = parser.parse_args()
    try:
        code = main(args.yes, args.validate_existing)
    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    sys.exit(code)
