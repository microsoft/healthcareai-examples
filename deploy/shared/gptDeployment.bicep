// GPT Model Deployment module
// This module deploys GPT models to an existing AI Services account

targetScope = 'resourceGroup'

@description('Name of the AI Services account')
param aiServicesName string

@description('GPT model name and version (e.g., "gpt-4o;2024-08-06")')
param gptModel string

@description('Tokens per minute capacity for the model (in thousands)')
param gptModelCapacity int = 100

@description('SKU name for the deployment')
@allowed(['GlobalStandard', 'Standard'])
param skuName string = 'GlobalStandard'

@description('Tags to apply to all resources')
param tags object = {}

// Extract model name and version from the gptModel parameter
var modelParts = split(gptModel, ';')
var modelName = modelParts[0]
var modelVersion = length(modelParts) > 1 ? modelParts[1] : '2024-08-06'

// Reference the existing AI Services account
resource aiServices 'Microsoft.CognitiveServices/accounts@2024-10-01' existing = {
  name: aiServicesName
}

// Deploy the GPT model if gptModel is not empty
resource gptDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = if (!empty(gptModel)) {
  parent: aiServices
  name: modelName
  properties: {
    model: {
      format: 'OpenAI'
      name: modelName
      version: modelVersion
    }
  }
  sku: {
    name: skuName
    capacity: gptModelCapacity
  }
  tags: tags
}

// Outputs
output deploymentName string = !empty(gptModel) ? gptDeployment.name : ''
output modelName string = !empty(gptModel) ? modelName : ''
output modelVersion string = !empty(gptModel) ? modelVersion : ''
output endpoint string = aiServices.properties.endpoint
