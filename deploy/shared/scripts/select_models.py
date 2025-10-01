#!/usr/bin/env python3
import sys
import json
import traceback
from utils import ensure_azd_env, set_azd_env_value, load_models

# Parse and validate selection using a function
def parse_and_validate_selection(selection, available_models_len):
    try:
        indices = [int(x.strip()) for x in selection.split(",") if x.strip()]
    except Exception:
        raise ValueError(
            "Invalid input. Please enter numbers separated by commas, or '*'."
        )
    if not indices or any(i < 1 or i > available_models_len for i in indices):
        raise ValueError(
            f"Invalid selection. Indices must be between 1 and {available_models_len}"
        )
    return indices


def main():
    # Ensure azd environment is active
    ensure_azd_env()
    # Load model definitions
    models = load_models()
    if not models:
        raise ValueError("No models found in models configuration.")

    # Build and print available models in one loop
    available_models = []
    print("Available models:")
    for idx, model in enumerate(models, 1):
        name = model.get("name", "")
        if not name:
            continue
        deployment = model.get("deployment", {})
        instance_type = deployment.get("instanceType", "<not set>")
        instance_count = deployment.get("instanceCount", "<not set>")
        available_models.append((name, instance_type, instance_count))
        print(f"  {len(available_models)}: {name}: {instance_type} x {instance_count}")
    if not available_models:
        raise ValueError("No valid models found in models configuration..")
    print()
    print(
        "Enter a comma-separated list of model numbers to deploy (e.g. 1,3,4), or '*' to deploy all:"
    )
    selection = input("Models to deploy: ").strip()
    if selection == "*":
        print("Deploying all models.")
        set_azd_env_value("modelFilter", "[]")
        return 0

    indices = parse_and_validate_selection(selection, len(available_models))
    selected_names = [available_models[i - 1][0] for i in indices]
    print(f"Selected models: {selected_names}")

    filter_json = json.dumps(selected_names)
    set_azd_env_value("modelFilter", filter_json)
    print("Set modelFilter in azd environment.")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    sys.exit(code)
