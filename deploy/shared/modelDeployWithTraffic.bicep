targetScope = 'resourceGroup'

@description('Azure ML workspace name')
param workspaceName string
@description('Azure region for deployment')
param location string
@description('Model name in Azure ML registry')
param modelName string
@description('Unique endpoint name')
param endpointName string = ''
@description('Deployment name')
param deploymentName string = ''
@description('Full AzureML model ID URI (e.g. azureml://registries/azureml/models/ModelName/versions/Version)')
param modelId string
@description('VM SKU for the deployment')
param instanceType string = 'Standard_NC4as_T4_v3'
@description('Number of instances to deploy')
param instanceCount int = 1
@description('Tags to apply to resources')
param tags object = {}
@description('Request settings for the deployment (object)')
param requestSettings object = {}
@description('Liveness probe settings for the deployment (object)')
param livenessProbe object = {}
@description('Environment variables for the deployment (object)')
param environmentVariables object = {}
@description('Overrideable unique suffix')
param uniqueSuffix string = ''

// Deploy endpoint and deployment (no traffic update)
module deploy 'mlModelEndpoint.bicep' = {
  name: 'deploy-${modelName}'
  params: {
    location: location
    workspaceName: workspaceName
    modelName: modelName
    endpointName: endpointName
    deploymentName: deploymentName
    modelId: modelId
    instanceType: instanceType
    instanceCount: instanceCount
    tags: tags
    requestSettings: requestSettings
    livenessProbe: livenessProbe
    environmentVariables: environmentVariables
    uniqueSuffix: uniqueSuffix
    setTraffic: false
  }
}

// Update traffic to 100% for this deployment
module updateTraffic 'mlModelEndpoint.bicep' = {
  name: 'update-traffic-${modelName}'
  params: {
    location: location
    workspaceName: workspaceName
    modelName: modelName
    endpointName: endpointName
    deploymentName: deploymentName
    modelId: modelId
    instanceType: instanceType
    instanceCount: instanceCount
    tags: tags
    requestSettings: requestSettings
    livenessProbe: livenessProbe
    environmentVariables: environmentVariables
    uniqueSuffix: uniqueSuffix
    setTraffic: true
  }
  dependsOn: [deploy]
}

output endpointName string = deploy.outputs.endpointName
output deploymentName string = deploy.outputs.deploymentName
output scoringUri string = deploy.outputs.scoringUri
output endpointId string = deploy.outputs.endpointId
