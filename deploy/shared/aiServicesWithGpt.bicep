// aiServicesWithGpt.bicep
// Combined module for AI Services and GPT deployment
// This module creates AI Services and GPT model, using externally provided infrastructure

@description('Azure region for deployment')
param location string

@description('Name of the AI Services account')
param aiServicesName string

@description('Tags to apply to all resources')
param tags object = {}

@description('Identities to grant access to the AI Services account')
param grantAccessTo array = []

@description('Additional managed identities to assign to the AI Services account')
param additionalIdentities array = []

@description('GPT model name and version to deploy (format: "model;version")')
param gptModel string

@description('Tokens per minute capacity for the model. Units of 1000 (capacity = 50 means 50K tokens per minute)')
param gptModelCapacity int = 50

// Note: This module only creates AI Services and GPT deployment
// Infrastructure connections are handled by the AI Hub module

// Create AI Services account
module aiServices './aiServices.bicep' = {
  name: 'ai-services'
  params: {
    location: location
    aiServicesName: aiServicesName
    tags: tags
    grantAccessTo: grantAccessTo
    additionalIdentities: additionalIdentities
  }
}

// Deploy GPT model to the AI Services account
module gptDeployment './gptDeployment.bicep' = {
  name: 'gpt-deployment'
  params: {
    aiServicesName: aiServices.outputs.aiServicesName
    gptModel: gptModel
    gptModelCapacity: gptModelCapacity
    tags: tags
  }
}

// Outputs
output aiServicesName string = aiServices.outputs.aiServicesName
output aiServicesEndpoint string = aiServices.outputs.aiServicesEndpoint
output aiServicesId string = aiServices.outputs.aiServicesId

output gptEndpoint string = gptDeployment.outputs.endpoint
output gptDeploymentName string = gptDeployment.outputs.deploymentName
output gptModelName string = gptDeployment.outputs.modelName
output gptModelVersion string = gptDeployment.outputs.modelVersion
output gptInferenceUri string = gptDeployment.outputs.inferenceUri
