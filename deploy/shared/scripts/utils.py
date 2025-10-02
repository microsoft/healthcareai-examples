import subprocess
import json
from pathlib import Path
import re

MODELS_USER_JSON = Path(__file__).parent.parent / "models.json"
MODELS_DEFAULT_JSON = Path(__file__).parent.parent / "modelsDefault.json"

REPO_ROOT = Path(__file__).parents[3]
REPO_ENV_FILE = REPO_ROOT / ".env"
REPO_EXAMPLE_ENV_FILE = REPO_ROOT / "env.example"

MODEL_FILTER_ENV_VAR = "HLS_MODEL_FILTER"

# ANSI colors for better readability
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
END = "\033[0m"

CMD_ECHO_ENABLED = True

def run_shell(cmd, capture_output=True, text=True, check=False, echo=False, shell=True):
    """
    Run a shell command using subprocess.
    """
    if echo or CMD_ECHO_ENABLED:
        cmd_str = ' '.join(cmd) if isinstance(cmd, list) else cmd
        print(f"{CYAN}Running: {cmd_str}{END}")
    
    return subprocess.run(
        cmd,
        capture_output=capture_output,
        text=text,
        check=check,
        shell=shell
    )


def get_model_filter():
    val = get_azd_env_value(MODEL_FILTER_ENV_VAR)
    if not val:
        return []
    return [item.strip() for item in val.split(",") if item.strip()]


def get_azd_env_value(key, default=None):
    result = run_shell(
        ["azd", "env", "get-value", key]
    )
    if result.returncode != 0 or not result.stdout.strip():
        return default
    return result.stdout.strip().strip('"')


def set_azd_env_value(key, value):
    result = run_shell(["azd", "env", "set", key, value])
    return result.returncode == 0


def load_azd_env_vars():
    """
    Load all AZD environment variables by invoking `azd env get-values`.
    """
    # `azd env get-values` outputs JSON of all key/value pairs
    result = run_shell(
        ["azd", "env", "get-values", "--output", "json"],
        check=True
    )
    return json.loads(result.stdout)


def parse_endpoints(endpoints_str):
    """Parse a JSON string of endpoints, raising ValueError on parse errors."""
    if not endpoints_str:
        return []
    try:
        cleaned = re.sub(r'^"|"$', "", endpoints_str).replace('\\"', '"')
        return json.loads(cleaned)
    except Exception as e:
        raise ValueError(f"Failed to parse endpoints JSON: {e}")


def ensure_azd_env():
    """
    Ensure an azd environment is active. Returns the environment name or raises RuntimeError if none.
    """
    env_name = get_azd_env_value("AZURE_ENV_NAME")
    if not env_name:
        raise RuntimeError(
            "No active azd environment detected. "
            "Please create (azd env new <env>) or select (azd env select <env>) an environment."
        )
    return env_name


def load_models():
    """Load models from JSON, returning a list of model dicts."""
    
    # Try to load models.json (user override file)
    models_path = Path(MODELS_USER_JSON)
    if not models_path.exists():
        raise FileNotFoundError(f"models.json not found at {models_path}")
    
    models_text = models_path.read_text().strip()
    
    # If models.json is empty or just empty array, use defaults
    if not models_text or models_text == "[]":
        default_path = Path(MODELS_DEFAULT_JSON)
        if not default_path.exists():
            raise FileNotFoundError(f"modelsDefault.json not found at {default_path}")
        models_text = default_path.read_text().strip()
    
    data = json.loads(models_text)
    
    # Validate structure
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return v
        raise ValueError("No model list found in JSON.")
    if isinstance(data, list):
        return data
    raise ValueError("models JSON improperly formatted.")


def get_ml_workspace(name: str, resource_group: str, subscription: str) -> dict:
    """
    Returns the Azure ML workspace object using Azure CLI, or raises RuntimeError if not found.
    """
    try:
        cmd = [
            "az",
            "ml",
            "workspace",
            "show",
            "--name",
            name,
            "--resource-group",
            resource_group,
            "--subscription",
            subscription,
            "--output",
            "json",
        ]

        result = run_shell(cmd, check=True)
        ws_data = json.loads(result.stdout)

        return {
            "location": ws_data.get("location"),
            "resourceGroup": ws_data.get("resource_group"),
            "id": ws_data.get("id"),
            "name": ws_data.get("name"),
        }

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else str(e)
        raise RuntimeError(
            f"Failed to retrieve workspace '{name}' in RG '{resource_group}': {error_msg}"
        )
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse Azure CLI response: {e}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to retrieve workspace '{name}' in RG '{resource_group}': {e}"
        )


def get_openai_api_key(ai_services_name: str, resource_group: str) -> str:
    """
    Retrieve the OpenAI API key for an AI Services resource using Azure CLI.

    Args:
        ai_services_name: Name of the Azure AI Services resource
        resource_group: Name of the resource group containing the AI Services

    Returns:
        The primary API key for the AI Services resource

    Raises:
        RuntimeError: If the API key retrieval fails
    """
    try:
        cmd = [
            "az",
            "cognitiveservices",
            "account",
            "keys",
            "list",
            "--name",
            ai_services_name,
            "--resource-group",
            resource_group,
            "--query",
            "key1",
            "--output",
            "tsv",
        ]

        result = run_shell(cmd, check=True)
        api_key = result.stdout.strip()

        if not api_key:
            raise RuntimeError("API key retrieval returned empty result")

        return api_key

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Azure CLI command failed: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve API key: {e}")


def detect_deployment_type():
    """Detect if we're in a 'fresh' or 'existing' deployment context."""
    # Start from current directory and walk up looking for azure.yaml
    current_dir = Path.cwd()

    azure_yaml = current_dir / "azure.yaml"
    if not azure_yaml.exists():
        raise RuntimeError(
            "This script should not be run directly in the root directory. "
            "Please run it from within the deploy/fresh or deploy/existing directories."
        )

    # Check if we're in or under deploy/fresh or deploy/existing
    for parent in [current_dir] + list(current_dir.parents):
        if parent.name in ["fresh", "existing"] and (parent.parent / "shared").exists():
            return parent.name

        # Also check if azure.yaml exists and we can infer from path
        azure_yaml = parent / "azure.yaml"
        if azure_yaml.exists():
            if "fresh" in str(parent):
                return "fresh"
            elif "existing" in str(parent):
                return "existing"

    return "unknown"
