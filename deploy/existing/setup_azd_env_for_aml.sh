#!/bin/bash
# Exit on error, undefined variables, and pipe failures
set -euo pipefail

#  NOTE: azd down IS NOT FUNCTIONAL for existing deployments
#
#  This script configures azd to deploy into your CURRENT AzureML workspace.
#  The 'azd down' command does not work for existing deployments and will not
#  delete any resources. Use '../shared/scripts/cleanup.py' to clean up
#  deployed resources when finished.
#
#  Only use this environment for deploying endpoints. For cleanup, use the
#  cleanup.py script provided in the shared/scripts folder.

# Setup color variables if supported
if command -v tput &>/dev/null; then
  RED=$(tput setaf 1)
  GREEN=$(tput setaf 2)
  YELLOW=$(tput setaf 3)
  BLUE=$(tput setaf 4)
  BOLD=$(tput bold)
  RESET=$(tput sgr0)
else
  RED=""; GREEN=""; YELLOW=""; BLUE=""; BOLD=""; RESET=""
fi

# Script to configure the current azd environment for deploying into the current AzureML workspace (when running from an AzureML compute instance)
# Usage: ./setup_azd_env_for_aml.sh

# Get the current azd environment
CURRENT_ENV=$(azd env get-value AZURE_ENV_NAME 2>/dev/null | grep -v "^ERROR:" || echo "")

if [ -z "$CURRENT_ENV" ]; then
    echo "Error: No current azd environment found. Please run 'azd env select <env-name>' first or create a new environment with 'azd env new <env-name>'."
    exit 1
fi

ENV_NAME="$CURRENT_ENV"
echo "Using current azd environment: $ENV_NAME"

# Check if running on AzureML compute instance
if [ "${APPSETTING_WEBSITE_SITE_NAME}" != "AMLComputeInstance" ]; then
    echo "This script is intended to be run from within an AzureML compute instance."
    exit 1
fi

echo "Detected AzureML compute instance environment."

WORKSPACE_NAME="${CI_WORKSPACE}"
RESOURCE_GROUP="${CI_RESOURCE_GROUP}"

if [ -z "$WORKSPACE_NAME" ] || [ -z "$RESOURCE_GROUP" ]; then
    echo "Error: Could not detect workspace info from AzureML compute instance."
    exit 1
fi

echo "Auto-detected workspace: $WORKSPACE_NAME in resource group: $RESOURCE_GROUP"

# We're using the current environment, no need to create or select

# Function to set azd environment variable with echo
azd_env_set() {
    echo "Setting $1 = $2"
    azd env set "$1" "$2"
}

# Set azd environment variables using helper
azd_env_set AZUREML_WORKSPACE_NAME "$WORKSPACE_NAME"
azd_env_set AZURE_RESOURCE_GROUP "$RESOURCE_GROUP"

# Get subscription ID
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
if [ -z "$SUBSCRIPTION_ID" ]; then
    echo "Error: Failed to retrieve subscription ID. Please ensure you are logged into Azure CLI."
    exit 1
fi
if [[ ! "$SUBSCRIPTION_ID" =~ ^[a-zA-Z0-9-]+$ ]]; then
  echo "Error: Invalid subscription ID format '$SUBSCRIPTION_ID'. Azure subscription IDs should be alphanumeric, may include dashes (e.g., 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx')."
  exit 1
fi
azd_env_set AZURE_SUBSCRIPTION_ID "$SUBSCRIPTION_ID"

# Set location from AzureML workspace
LOCATION=$(az ml workspace show --name "$WORKSPACE_NAME" --resource-group "$RESOURCE_GROUP" --subscription "$SUBSCRIPTION_ID" --query location -o tsv)
if [ -z "$LOCATION" ]; then
    echo "Error: Failed to retrieve workspace location. Please verify the workspace exists and you have access."
    exit 1
fi
# Validate location format (Azure locations are alphanumeric, no spaces)
if [[ ! "$LOCATION" =~ ^[a-zA-Z0-9]+$ ]]; then
    echo "Error: Invalid location format '$LOCATION'. Azure locations should be alphanumeric with no spaces (e.g., 'eastus', 'westeurope')."
    exit 1
fi
azd_env_set AZURE_LOCATION "$LOCATION"

# Success info
echo ""
echo "${GREEN}Environment '$ENV_NAME' successfully configured for AzureML workspace '$WORKSPACE_NAME'.${RESET}"
echo "${GREEN}Run 'azd up' to deploy endpoints into this workspace.${RESET}"
echo "${GREEN}Or run '../shared/scripts/select_models.py' to choose specific models before deployment.${RESET}"
