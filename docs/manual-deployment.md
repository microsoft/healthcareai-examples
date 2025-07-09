# Manual Deployment Guide

For users who prefer manual deployment, you can deploy healthcare AI models using either the Azure portal or Python SDK.

## Prerequisites

- Azure subscription with sufficient quota for the models you want to deploy

## Step 1: Create Azure ML Workspace (if needed)

If you don't have an existing Azure ML workspace, create one first: [Azure ML workspace creation guide](https://learn.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources)


## Step 2: Deploy Healthcare AI Models

Choose one of the following deployment methods:

### Option 1: Azure Portal Deployment

Follow the official Microsoft documentation to deploy the healthcare AI models you need:

- **[Overview of Foundation models for healthcare AI](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/healthcare-ai/healthcare-ai-models)** - General overview and concepts
- **[MedImageInsight](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/healthcare-ai/deploy-medimageinsight)** - Medical image analysis deployment guide
- **[CXRReportGen](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/healthcare-ai/deploy-cxrreportgen)** - Chest X-ray report generation deployment guide  
- **[MedImageParse](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/healthcare-ai/deploy-medimageparse?tabs=medimageparse)** - Medical image segmentation deployment guide

These guides provide step-by-step instructions for deploying models through Azure AI Foundry Studio, including SKU recommendations and configuration options.

### Option 2: Python SDK Deployment

For programmatic deployment in existing Azure ML workspaces, use our deployment notebooks:

* [MedImageInsight deployment](https://aka.ms/healthcare-ai-examples-mi2-deploy)
* [MedImageParse deployment](https://aka.ms/healthcare-ai-examples-mip-deploy)
* [CXRReportGen deployment](https://aka.ms/healthcare-ai-examples-cxr-deploy)

## Step 3: Get Endpoint Resource ID

After deployment completes, you'll need the Azure resource ID for each endpoint. There are three ways to obtain this:

### Option 1: Azure CLI (Recommended)

Use the Azure CLI to list and get endpoint details:

```bash
# List all endpoints in your workspace
az ml online-endpoint list --resource-group {your-resource-group} --workspace-name {your-workspace}

# Get specific endpoint details (including resource ID)
az ml online-endpoint show --name {your-endpoint-name} --resource-group {your-resource-group} --workspace-name {your-workspace}
```

The resource ID will be in the `id` field of the output.

### Option 2: Azure Portal

1. Go to the [Azure Portal](https://portal.azure.com)
2. Navigate to your deployed endpoint resource
3. Copy the resource ID from the browser URL

The URL will look like:
```
https://portal.azure.com/#@yourtenant.onmicrosoft.com/resource/subscriptions/12345678-1234-1234-1234-123456789abc/resourceGroups/your-resource-group/providers/Microsoft.MachineLearningServices/workspaces/your-workspace/onlineEndpoints/your-endpoint-name/overview
```

The resource ID is the part after `/resource/`:
```
/subscriptions/12345678-1234-1234-1234-123456789abc/resourceGroups/your-resource-group/providers/Microsoft.MachineLearningServices/workspaces/your-workspace/onlineEndpoints/your-endpoint-name
```

### Option 3: Python SDK

If you deployed using the Python SDK, you can get the resource ID programmatically:

```python
# After creating your endpoint object
print(f"Endpoint resource ID: {endpoint.id}")
```

## Step 4: Update Environment Variables

Create a `.env` file for environment variables:

```sh
cp env.example .env
```

Add the endpoint resource IDs to your `.env` file:

```bash
# Replace with your actual endpoint resource IDs (with leading slash)
MI2_MODEL_ENDPOINT=/subscriptions/{your-sub-id}/resourceGroups/{your-rg}/providers/Microsoft.MachineLearningServices/workspaces/{your-workspace}/onlineEndpoints/{your-medimageinsight-endpoint}
MIP_MODEL_ENDPOINT=/subscriptions/{your-sub-id}/resourceGroups/{your-rg}/providers/Microsoft.MachineLearningServices/workspaces/{your-workspace}/onlineEndpoints/{your-medimageparse-endpoint}
CXRREPORTGEN_MODEL_ENDPOINT=/subscriptions/{your-sub-id}/resourceGroups/{your-rg}/providers/Microsoft.MachineLearningServices/workspaces/{your-workspace}/onlineEndpoints/{your-cxrreportgen-endpoint}
```

**Note**: Use the full resource ID path (with the leading slash) as shown above. Replace the placeholder values in curly braces with your actual resource names and IDs. See `env.example` for more examples and detailed formatting instructions.

## Next Steps

Once deployed, return to the main README and continue with [Step 4: Setup your local environment](../README.md#step-4-setup-your-local-environment)

## Troubleshooting

For additional troubleshooting, see the main [Deployment Guide](deployment-guide.md#troubleshooting).
