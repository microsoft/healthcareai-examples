targetScope = 'resourceGroup'

// -----------------------------------------------------------------------------
// Required parameters
// -----------------------------------------------------------------------------
@description('Azure ML workspace name')
param workspaceName string

@description('Model name in Azure ML registry')
param modelName string

@description('Unique endpoint name')
param endpointName string = ''

@description('Deployment name')
param deploymentName string = ''

@description('Azure region for deployment')
param location string

@description('Full AzureML model ID URI (e.g. azureml://registries/azureml/models/ModelName/versions/Version)')
param modelId string

// -----------------------------------------------------------------------------
// Optional parameters with defaults
// -----------------------------------------------------------------------------
@description('VM SKU for the deployment')
param instanceType string = 'Standard_NC4as_T4_v3'

@description('Number of instances to deploy')
param instanceCount int = 1

@description('Maximum concurrent requests per instance')
param maxConcurrentRequestsPerInstance int = 1

@description('Tags to apply to resources')
param tags object = {
  Repo: 'microsoft/healthcareai-examples-pr'
  Environment: 'azd'
  DeployedBy: 'azd'
}

@description('Overrideable unique suffix (provided by root through effective computation)')
param uniqueSuffix string = ''

// Set endpointName to the provided value, or fallback to toLower(modelName)-suffix if not provided
var effectiveEndpointName = !empty(endpointName) ? endpointName : toLower(format('{0}-{1}', modelName, uniqueSuffix))

var modelIdParts = split(modelId, '/')
var parsedModelName = modelIdParts[5]
var parsedModelVersion = modelIdParts[7]
var effectiveDeploymentName = !empty(deploymentName) ? deploymentName : toLower(format('{0}-v{1}', parsedModelName, parsedModelVersion))

@description('Request settings for the deployment (object with keys like requestTimeout, maxConcurrentRequestsPerInstance, etc.)')
param requestSettings object = {}
var defaultRequestSettings = {
  requestTimeout: 'PT1M30S'
  maxConcurrentRequestsPerInstance: maxConcurrentRequestsPerInstance
}
var effectiveRequestSettings = union(defaultRequestSettings, requestSettings)

@description('Liveness probe settings for the deployment (object with keys like initialDelay, etc.)')
param livenessProbe object = {}
var defaultLivenessProbe = {
  initialDelay: 'PT10M'
}
var effectiveLivenessProbe = union(defaultLivenessProbe, livenessProbe)

// WORKER_COUNT: use environmentVariables.WORKER_COUNT if present, otherwise default to requestSettings_maxConcurrentRequestsPerInstance
@description('Environment variables for the deployment (object, e.g. { WORKER_COUNT: 3 })')
param environmentVariables object = {}
var defaultEnvironmentVariables = {
  WORKER_COUNT: string(effectiveRequestSettings.maxConcurrentRequestsPerInstance)
}
var effectiveEnvironmentVariables = union(defaultEnvironmentVariables, environmentVariables)

param setTraffic bool = false
var effectiveTraffic = setTraffic ? {'${effectiveDeploymentName}': 100} : null

// -----------------------------------------------------------------------------
// Resources
// -----------------------------------------------------------------------------

// Reference to existing Azure ML workspace
resource workspace 'Microsoft.MachineLearningServices/workspaces@2024-10-01' existing = {
  name: workspaceName
}

// Create the online endpoint
resource onlineEndpoint 'Microsoft.MachineLearningServices/workspaces/onlineEndpoints@2024-10-01' = {
  parent: workspace
  name: effectiveEndpointName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    authMode: 'key'
    publicNetworkAccess: 'Enabled'
    traffic: effectiveTraffic
  }
  tags: union(tags, {
    Model: modelName
  })
}

// Create the model deployment
resource modelDeployment 'Microsoft.MachineLearningServices/workspaces/onlineEndpoints/deployments@2024-10-01' = {
  parent: onlineEndpoint
  name: effectiveDeploymentName
  location: location
  sku: {
    name: 'Default'
    tier: 'Standard'
    capacity: instanceCount
  }
  properties: {
    endpointComputeType: 'Managed'
    model: modelId
    instanceType: instanceType
    scaleSettings: {
      scaleType: 'Default'
    }
    requestSettings: effectiveRequestSettings
    environmentVariables: effectiveEnvironmentVariables
    livenessProbe: effectiveLivenessProbe
    appInsightsEnabled: true

  }
  tags: union(tags, {
    Model: modelName
  })
}


// -----------------------------------------------------------------------------
// Outputs
// -----------------------------------------------------------------------------
output endpointName string = onlineEndpoint.name
output deploymentName string = modelDeployment.name
output scoringUri string = onlineEndpoint.properties.scoringUri
output endpointId string = onlineEndpoint.id
