// Resource-group-scoped template
// Creates a fresh AML workspace with all supporting infrastructure

targetScope = 'resourceGroup'

// ============================================================================
// PARAMETERS - Basic Configuration
// ============================================================================

@description('Name of the environment')
param environmentName string

@description('Name of the Azure ML workspace')
param workspaceName string = 'mlw-${environmentName}'

@description('Tags to apply to all resources')
param tags object = {}

@description('Comma-separated list of model names to include (filter)')
param modelFilterString string = ''

@description('Azure region for deployment')
param location string = resourceGroup().location

@description('Unique suffix for resource naming (overrideable for consistency)')
param uniqueSuffix string = ''

// ============================================================================
// PARAMETERS - GPT Configuration
// ============================================================================

@description('Gen AI model name and version to deploy, leave empty to skip')
@allowed(['','gpt-4o;2024-08-06', 'gpt-4.1;2025-04-14'])
param gptModel string

@description('Tokens per minute capacity for the model. Units of 1000 (capacity = 100 means 100K tokens per minute)')
param gptModelCapacity int = 100

@description('Azure region for GPT deployment (can be different from main location)')
param gptDeploymentLocation string = ''

// ============================================================================
// PARAMETERS - Access Control
// ============================================================================

@description('Principal ID to grant access to the AI services. Leave empty to skip')
param myPrincipalId string = ''

@description('Current principal type being used')
@allowed(['User', 'ServicePrincipal'])
param myPrincipalType string

// ============================================================================
// PARAMETERS - Resource Names (Optional Overrides)
// ============================================================================

// AI Services configurations
@description('Name of the AI Services account. Automatically generated if left blank')
param aiServicesName string = ''

@description('Name of the Storage Account. Automatically generated if left blank')
param storageName string = ''

@description('Name of the Key Vault. Automatically generated if left blank')
param keyVaultName string = ''

@description('Name of the Container Registry. Automatically generated if left blank')
param containerRegistryName string = ''

@description('Allow shared key access to storage account (AZURE_STORAGE_ALLOW_ACCESS_KEY environment variable, default: false)')
param allowSharedKeyAccess bool = false

// ============================================================================
// VARIABLES - Configuration and Naming
// ============================================================================
var effectiveUniqueSuffix = empty(uniqueSuffix) ? substring(uniqueString(resourceGroup().id), 0, 6) : uniqueSuffix
var effectiveGptLocation = empty(gptDeploymentLocation) ? location : gptDeploymentLocation

var environmentNameTrunc = substring(((replace(replace(environmentName, '-', ''), '_', ''))),0,6)



// Default tags to apply to all resources
var defaultTags = {
  'azd-env-name': environmentName
  'azd-deployment-type': 'fresh'
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
  storage: !empty(storageName) ? storageName : '${abbrs.storageStorageAccounts}${environmentNameTrunc}${effectiveUniqueSuffix}'
  keyVault: !empty(keyVaultName) ? keyVaultName : '${abbrs.keyVaultVaults}${environmentNameTrunc}${effectiveUniqueSuffix}'
  containerRegistry: !empty(containerRegistryName) ? containerRegistryName : '${abbrs.containerRegistryRegistries}${environmentNameTrunc}${effectiveUniqueSuffix}'
  aiServices: !empty(aiServicesName) ? aiServicesName : '${abbrs.cognitiveServicesAccounts}${environmentName}-${effectiveUniqueSuffix}'
}

// Create AML workspace with all its dependencies
module workspace '../../shared/amlWorkspace.bicep' = {
  name: 'aml-workspace'
  scope: resourceGroup()
  params: {
    location: location
    workspaceName: workspaceName
    tags: tagsUpdated
    storageAccountName: names.storage
    keyVaultName: names.keyVault
    containerRegistryName: names.containerRegistry
    allowSharedKeyAccess: allowSharedKeyAccess
    grantAccessTo: [
      {
        id: myPrincipalId
        type: myPrincipalType
      }
    ]
    additionalIdentities: []
  }
}

// Deploy GPT services (AI Services + GPT model) if specified
module gptServices '../../shared/aiServicesWithGpt.bicep' = if (!empty(gptModel)) {
  name: 'gpt-services'
  scope: resourceGroup()
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

// Deploy model endpoints into this resource group
module modelDeploy '../../shared/deployModel.bicep' = {
  name: 'deploy-models'
  scope: resourceGroup()
  params: {
    workspaceName: workspace.outputs.workspaceName
    location: location
    tags: tagsUpdated
    modelFilterString: modelFilterString
    uniqueSuffix: effectiveUniqueSuffix
  }
}

// Outputs
output AZURE_SUBSCRIPTION_ID string           = subscription().subscriptionId
output AZURE_RESOURCE_GROUP_ID string         = resourceGroup().id
output AZURE_RESOURCE_GROUP string            = resourceGroup().name
output AZUREML_WORKSPACE_ID string            = workspace.outputs.workspaceId
output AZUREML_WORKSPACE_NAME string          = workspace.outputs.workspaceName

output AZUREML_WORKSPACE_LOCATION string       = workspace.outputs.workspaceLocation
output AZUREML_STORAGE_ACCOUNT_ENDPOINT string = workspace.outputs.storageAccountBlobEndpoint
output AZUREML_KEY_VAULT_ENDPOINT string       = workspace.outputs.keyVaultEndpoint

output UNIQUE_SUFFIX string                   = effectiveUniqueSuffix
output HLS_MODEL_ENDPOINTS array              = modelDeploy.outputs.endpoints

// GPT deployment outputs (conditional)
output AZURE_OPENAI_ENDPOINT string      = !empty(gptModel) ? gptServices.outputs.gptEndpoint : ''
output AZURE_OPENAI_MODEL_NAME string         = !empty(gptModel) ? gptServices.outputs.gptModelName : ''
output AZURE_AI_SERVICES_NAME string     = !empty(gptModel) ? gptServices.outputs.aiServicesName : ''
