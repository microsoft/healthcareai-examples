# Deploy to Existing Environment

This deployment option deploys AI model endpoints into your existing Azure ML workspace. Use this option when you already have a workspace set up and want to add healthcare AI models to it.

## Prerequisites

- Azure CLI and Azure Developer CLI (azd) installed
- Existing Azure ML workspace with appropriate permissions
- Contributor or equivalent role on the target resource group or subscription (sufficient for deploying to existing infrastructure)
- Required quota for the models you plan to deploy (see main README Step 1)


## What Gets Deployed

### Healthcare AI Models
- **Model Endpoints**: AI models deployed as managed online endpoints in your existing workspace
- **Compute Resources**: Dedicated compute instances for each model
- **No Infrastructure Changes**: Your existing workspace, storage, and other resources remain untouched

### GPT Integration (when enabled)
- **Azure AI Services**: New AI Services account in your existing resource group
- **GPT Model Endpoints**: Ready for use alongside healthcare AI models

> [!TIP]
> **Workspace Permissions**: Ensure you have appropriate permissions in the target workspace to create endpoints and compute resources.

## Deployment Steps

> [!NOTE]
> See [Deployment Configuration](../../docs/deployment-guide.md#deployment-configuration) for detailed setup options.

### 1. Navigate to Existing Deployment Directory

```bash
cd deploy/existing
```

### 2. Authenticate with Azure

```bash
az login                 # add -t <TENANT_ID> if needed
azd auth login           # add --tenant <TENANT_ID> if needed
```

### 3. Create and Configure Environment

```bash
# Create a new azd environment
azd env new <envName>

# Configure for existing workspace deployment
azd env set AZURE_RESOURCE_GROUP <your-existing-rg>
azd env set AZUREML_WORKSPACE_NAME <your-existing-workspace>
azd env set AZURE_LOCATION <workspace-location>
```

> [!IMPORTANT]
> **Azure ML Compute Instance**: If you're running from within an Azure ML compute instance, you can use our helper script to automatically detect your current workspace settings:
> 
> ```bash
> # Auto-configure environment from current AML compute instance
> ./setup_azd_env_for_aml.sh
> 
> # Then continue with step 4 below
> ```
> 
> This script automatically detects your current workspace name, resource group, subscription ID, and workspace location.

### 4. Optional: Configure GPT Model Deployment

See [GPT Model Configuration](../../docs/deployment-guide.md#gpt-model-configuration) in the deployment guide for GPT model setup options.

### 5. Optional: Select Specific Healthcare AI Models

See [Model Selection](../../docs/deployment-guide.md#model-selection) in the deployment guide for filtering which models to deploy.

### 6. Deploy Models

```bash
azd up
```

This command will:
- Use your existing resource group and Azure ML workspace
- Deploy the selected healthcare AI model endpoints
- **If GPT model specified**: Deploy Azure AI Services and GPT model in the same resource group
- Configure your `.env` file with connection details

## Environment Variables

After successful deployment, your root level `.env` file should contain:

```bash
# Healthcare AI model endpoints
MI2_MODEL_ENDPOINT=<medimageinsight-endpoint-id>
MIP_MODEL_ENDPOINT=<medimageparse-endpoint-id>
CXRREPORTGEN_MODEL_ENDPOINT=<cxrreportgen-endpoint-id>

# GPT integration variables (if GPT model was deployed)
AZURE_OPENAI_ENDPOINT=<gpt-endpoint-uri>
AZURE_OPENAI_MODEL_NAME=<gpt-model-name>
AZURE_OPENAI_API_KEY=<api-key>
```
## Next Steps
After successful deployment, change back to the root directory:

```bash
cd ../../
```

Then return to the main README and continue with [Step 4: Setup your local environment](../../README.md#step-4-setup-your-local-environment) to install the Healthcare AI Toolkit and explore the examples.
## Resource Cleanup

### Quick Cleanup - Model Deployments Only (Recommended)

To save costs by stopping expensive GPU compute resources while keeping your infrastructure:

```bash
# Delete model deployments only (they charge per hour)
python ../shared/scripts/cleanup.py

# Delete without confirmation
python ../shared/scripts/cleanup.py --yes
```
This removes only the model endpoint deployments that charge per hour, while keeping the infrastructure (workspace, storage, etc.) for future use. See [Resource Cleanup](../../docs/deployment-guide.md#resource-cleanup) in the deployment guide for more details.

### Complete Resource Cleanup

> [!IMPORTANT]
> **azd down Limitation**: The `azd down` command does not work for existing deployments due to the subscription-scoped template design.

#### Option 1: Delete Everything with Cleanup Script (Recommended)

```bash
# Delete all azd-tagged resources
python ../shared/scripts/cleanup.py --all --purge

# Delete everything without confirmation  
python ../shared/scripts/cleanup.py --all --purge --yes
```

#### Option 2: Manual Cleanup

If you prefer manual cleanup, all deployed resources are tagged with your environment name (`azd-env-name=<your-env-name>`).

**Azure CLI Method:**
```bash
# List resources first to see what will be deleted (replace 'your-env-name' with your actual environment name)
az resource list --tag "azd-env-name=your-env-name" --query "[].{Name:name, Type:type, ResourceGroup:resourceGroup, Location:location}" -o table

# Delete all resources with this tag
az resource delete --ids $(az resource list --tag "azd-env-name=your-env-name" --query "[].id" -o tsv)
```

**Azure Portal Method:**
- Navigate to your resource group in the Azure Portal
- Filter resources by the tag `azd-env-name` with your environment name
- Select and delete the resources you want to remove

## Important Notes



## Troubleshooting

For comprehensive troubleshooting including GPT deployment issues, see [Troubleshooting](../../docs/deployment-guide.md#troubleshooting) in the deployment guide.