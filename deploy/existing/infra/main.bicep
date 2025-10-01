// main-existing.bicep
// Subscription-scoped template
// Uses an existing resource group and AML workspace; no creation logic

targetScope = 'subscription'

@description('Name of the environment')
param environmentName string

@description('Name of the resource group to use')
param resourceGroupName string

@description('Name of the Azure ML workspace')
param workspaceName string

@description('Tags to apply to all resources')
param tags object = {}

@description('Comma-separated list of model names to include (filter)')
param modelFilterString string = ''

@description('JSON string containing models configuration (optional, defaults to values in models.json)')
param modelsJsonString string = ''

@description('Azure region for deployment')
param location string

@description('Unique suffix for resource naming (overrideable for consistency)')
param uniqueSuffix string = ''

@description('Gen AI model name and version to deploy, leave empty to skip')
@allowed(['','gpt-4o;2024-08-06', 'gpt-4.1;2025-04-14'])
param gptModel string

@description('Tokens per minute capacity for the model. Units of 1000 (capacity = 100 means 100K tokens per minute)')
param gptModelCapacity int = 50

@description('Azure region for GPT deployment (can be different from main location)')
param gptDeploymentLocation string = ''

@description('Principal ID to grant access to the AI services. Leave empty to skip')
param myPrincipalId string = ''

@description('Current principal type being used')
@allowed(['User', 'ServicePrincipal'])
param myPrincipalType string

// AI Services configurations
@description('Name of the AI Services account. Automatically generated if left blank')
param aiServicesName string = ''

// Variables for resource naming
var effectiveGptLocation = empty(gptDeploymentLocation) ? location : gptDeploymentLocation
var effectiveUniqueSuffix = empty(uniqueSuffix) ? substring(uniqueString('${subscription().id}/resourceGroups/${resourceGroupName}'), 0, 6) : uniqueSuffix

// Default tags to apply to all resources
var defaultTags = {
  'azd-env-name': environmentName
  'azd-deployment-type': 'existing'
  'azd-deployed-by': 'azd'
  'azd-id': effectiveUniqueSuffix
  Environment: 'Non-Prod'
}

// Merge user tags with default tags
var tagsUpdated = union(defaultTags, tags)

// Azure resource abbreviations
var abbrs = loadJsonContent('../../shared/abbreviations.json')

// Centralized resource names
var names = {
  aiServices: !empty(aiServicesName) ? aiServicesName : '${abbrs.cognitiveServicesAccounts}${environmentName}-${effectiveUniqueSuffix}'
}

// Reference existing AML workspace
resource existingWorkspace 'Microsoft.MachineLearningServices/workspaces@2024-10-01' existing = {
  scope: resourceGroup(resourceGroupName)
  name: workspaceName
}

// Deploy GPT services (AI Services + GPT model) if specified
module gptServices '../../shared/aiServicesWithGpt.bicep' = if (!empty(gptModel)) {
  name: 'gpt-services'
  scope: resourceGroup(resourceGroupName)
  params: {
    location: effectiveGptLocation
    aiServicesName: names.aiServices
    gptModel: gptModel
    gptModelCapacity: gptModelCapacity
    tags: tagsUpdated
    grantAccessTo: [
      {
        id: myPrincipalId
        type: myPrincipalType
      }
    ]
    additionalIdentities: []
  }
}

// Deploy model endpoints into the specified resource group
module modelDeploy '../../shared/deployModel.bicep' = {
  name: 'deploy-models'
  scope: resourceGroup(resourceGroupName)
  params: {
    workspaceName: workspaceName
    location: location
    tags: tagsUpdated
    modelFilterString: modelFilterString
    uniqueSuffix: effectiveUniqueSuffix
    modelsJsonString: modelsJsonString
  }
}

// Outputs
output AZURE_SUBSCRIPTION_ID string      = subscription().subscriptionId
output AZURE_RESOURCE_GROUP string       = resourceGroupName
output AZUREML_WORKSPACE_ID string       = existingWorkspace.id
output AZUREML_WORKSPACE_NAME string     = workspaceName
output HLS_MODEL_ENDPOINTS array         = modelDeploy.outputs.endpoints
output UNIQUE_SUFFIX string              = effectiveUniqueSuffix

// GPT deployment outputs (conditional)
output AZURE_OPENAI_ENDPOINT string      = !empty(gptModel) ? gptServices.outputs.gptEndpoint : ''
output AZURE_OPENAI_MODEL_NAME string         = !empty(gptModel) ? gptServices.outputs.gptModelName : ''
output AZURE_AI_SERVICES_NAME string     = !empty(gptModel) ? gptServices.outputs.aiServicesName : ''
