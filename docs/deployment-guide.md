# Healthcare AI Deployment Guide

This comprehensive guide covers all deployment methods for Healthcare AI models, including optional GPT model deployment.

## Quick Start

The fastest way to get started is with the Azure Developer CLI (azd), which automatically provisions all required infrastructure and deploys the models.

### Prerequisites

- [Azure Developer CLI](https://learn.microsoft.com/en-us/azure/developer/azure-developer-cli/install-azd) installed
- Azure subscription with sufficient quota
- Azure CLI logged in (`az login`)

### Basic Deployment (Healthcare AI Only)

```bash
# Clone and navigate to deployment folder
cd deploy/fresh

# Create new environment
azd env new healthcareai-examples-env

# Set your preferred Azure region
azd env set AZURE_LOCATION "eastus2"

# Deploy everything
azd up
```

This deploys all three healthcare AI models (MedImageInsight, MedImageParse, CXRReportGen) with a new Azure ML workspace.

### Output Environment Variables

After successful deployment, your root level `.env` file should contain:

```bash
# Healthcare AI model endpoints
MI2_MODEL_ENDPOINT=<medimageinsight-endpoint-id>
MIP_MODEL_ENDPOINT=<medimageparse-endpoint-id>
CXRREPORTGEN_MODEL_ENDPOINT=<cxrreportgen-endpoint-id>

# GPT integration variables (if GPT model was deployed)
AZURE_OPENAI_ENDPOINT=<gpt-endpoint-uri>
AZURE_OPENAI_API_KEY=<api-key>
```

## Deployment Configuration

Choose the deployment method that best fits your environment and requirements:

- **[Fresh Deployment](../deploy/fresh/README.md)** - Creates new resource group and Azure ML workspace
- **[Existing Workspace Deployment](../deploy/existing/README.md)** - Uses your existing Azure ML workspace

> [!NOTE]
> **Manual Deployment**: For users who prefer manual deployment, see the [Manual Deployment Guide](manual-deployment.md) which covers Azure Portal and Python SDK deployment methods.

### Model Selection

By default, all three healthcare AI models (MedImageInsight, MedImageParse, CXRReportGen) are deployed. You can optionally select specific models:

```bash
# Interactive model selection
python ../shared/scripts/select_models.py

# Or set via environment variable
azd env set HLS_MODEL_FILTER "medimageinsight,cxrreportgen"
```

### GPT Model Configuration

#### GPT Model Options

| Model | Model String | Recommended Capacity | Description |
|-------|-------------|---------------------|-------------|
| GPT-4o | `"gpt-4o;2024-08-06"` | 50-100K TPM | Latest multimodal model |
| GPT-4.1 | `"gpt-4.1;2025-04-14"` | 50-100K TPM | Advanced reasoning model |

#### Environment Variables

| Variable | Default | Description | Example Values |
|----------|---------|-------------|----------------|
| `AZURE_GPT_MODEL` | `""` (skip) | GPT model and version | `"gpt-4o;2024-08-06"`, `"gpt-4.1;2025-04-14"` |
| `AZURE_GPT_CAPACITY` | `"100"` | Tokens per minute (thousands) | `"100"`, `"200"` |
| `AZURE_GPT_LOCATION` | `""` (main location) | Deployment region | `"southcentralus"`, `"westus3"` |

#### Example Configurations

**Deploy GPT-4o with default capacity:**
```bash
azd env set AZURE_GPT_MODEL "gpt-4o;2024-08-06"
azd up
```

**Deploy GPT-4.1 with custom capacity:**
```bash
azd env set AZURE_GPT_MODEL "gpt-4.1;2025-04-14"
azd env set AZURE_GPT_CAPACITY "100"
azd up
```

**Deploy GPT in different region:**
```bash
azd env set AZURE_GPT_MODEL "gpt-4o;2024-08-06"
azd env set AZURE_GPT_LOCATION "southcentralus"
azd up
```

#### Tips for GPT Deployment

- **Quota**: Ensure you have Azure OpenAI quota in your target region before deployment
- **Capacity Planning**: Start with 50K TPM and adjust based on usage patterns
- **Region Selection**: Some GPT models may have better availability in specific regions
- **Integration Ready**: GPT endpoints work seamlessly with healthcare AI models for multimodal workflows

## Next Steps

Once deployed, return to the main README and continue with [Step 4: Setup your local environment](../README.md#step-4-setup-your-local-environment)

## Resource Cleanup

### Quick Cleanup - Model Deployments Only (Recommended)

To save costs by stopping expensive GPU compute resources while keeping your infrastructure:

```bash
# Delete only model endpoint deployments (they charge per hour)
python cleanup.py

# Delete without confirmation
python cleanup.py --yes
```

This removes only the model endpoint deployments that charge per hour while running, keeping the infrastructure (workspace, storage, etc.) for future use.

### Complete Resource Cleanup

For complete cleanup instructions specific to your deployment method:

- **Fresh deployments**: See [Fresh Deployment Cleanup](../deploy/fresh/README.md#resource-cleanup)
- **Existing deployments**: See [Existing Deployment Cleanup](../deploy/existing/README.md#resource-cleanup)

## Troubleshooting

### Azure Developer CLI Issues
- **Permission Issues**: Ensure your account has Contributor role on the subscription or resource group
- **Quota Issues**: Verify you have sufficient quota in your selected region

### GPT Deployment Issues
- **Quota**: Ensure you have Azure OpenAI quota in the target region
- **Region**: Try different regions if quota is unavailable
- **Model availability**: Verify the model version is available in your region

### Common Error Messages
- `"Insufficient quota"`: Request more quota in Azure portal for the specific VM family or OpenAI TPM
- `"Model not found"`: Check model name/version spelling and regional availability
- `"Region not supported"`: Try a different Azure region
- `"Permission denied"`: Verify you have Contributor access to the resource group or subscription

For manual deployment troubleshooting, see the [Manual Deployment Guide](manual-deployment.md#troubleshooting).