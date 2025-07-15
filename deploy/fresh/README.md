# Deploy to Fresh Environment

This deployment option creates a completely new Azure environment with a fresh resource group and Azure ML workspace. Use this option when you want isolated resources or are setting up a new development environment.

## Prerequisites

- Azure CLI and Azure Developer CLI (azd) installed
- **Owner** role OR **User Access Administrator + Contributor** roles on the Azure subscription (required for creating resource groups and configuring role-based access controls)
- Required quota for the models you plan to deploy (see main README Step 1)

> [!NOTE]
> **For Admins**: You can deploy on behalf of another user by setting `AZURE_PRINCIPAL_ID` to their Azure AD object ID. This grants the target user access to the deployed resources while you maintain deployment permissions.

## What Gets Created

### Core Infrastructure
- **Resource Group**: New resource group containing all resources
- **Azure ML Workspace**: Fresh workspace with system-assigned managed identity
- **Storage Account**: For workspace data and artifacts
- **Key Vault**: For secrets and keys management
- **Container Registry**: For model containers
- **Application Insights**: For monitoring and logging
- **Healthcare AI Model Endpoints**: Deployed models ready for inference

### GPT Integration (when enabled)
- **Azure AI Services**: Multi-service AI account for GPT models
- **GPT Model Endpoints**: Ready for use alongside healthcare AI models

> [!TIP]
> **Resource Naming**: Resources are created with unique names using a hash of the resource group ID to avoid naming conflicts.

## Deployment Steps

> [!NOTE]
> See [Deployment Configuration](../../docs/deployment-guide.md#deployment-configuration) for detailed setup options.

### 1. Navigate to Fresh Deployment Directory

```bash
cd deploy/fresh
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

# Set your Azure location (use a location where you have quota)
azd env set AZURE_LOCATION <location>
```

### 4. Optional: Select Specific Healthcare AI Models

See [Model Selection](../../docs/deployment-guide.md#model-selection) in the deployment guide for filtering which models to deploy.

### 5. Optional: Configure GPT Model Deployment

See [GPT Model Configuration](../../docs/deployment-guide.md#gpt-model-configuration) in the deployment guide for GPT model setup options.

### 6. Deploy Resources

```bash
azd up
```

This command will:
- Create a new resource group
- Deploy a new Azure ML workspace with associated resources (storage, key vault, container registry, etc.)
- Deploy the selected healthcare AI model endpoints
- **If GPT model specified**: Deploy Azure AI Services and GPT model
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

When you're done with the deployment and want to delete all resources:

```bash
azd down --purge
```
This removes the entire resource group and all contained resources.

> [!NOTE]
> **Known Issue**: `azd down --purge` will not actually purge the Azure ML workspace due to a known limitation. The workspace will be soft-deleted and can be recovered within the retention period.

## Troubleshooting

See [Troubleshooting](../../docs/deployment-guide.md#troubleshooting)