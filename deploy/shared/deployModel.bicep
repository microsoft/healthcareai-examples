targetScope = 'resourceGroup'

// -----------------------------------------------------------------------------
// Required parameters
// -----------------------------------------------------------------------------
@description('Azure ML workspace name')
param workspaceName string

@description('Azure region for deployment')
param location string

// -----------------------------------------------------------------------------
// Optional parameters with defaults
// -----------------------------------------------------------------------------
@description('Tags to apply to resources')
param tags object = {}

@description('Comma-separated list of model names to include (filter)')
param modelFilterString string = ''

@description('Overrideable unique suffix to pass to submodule')
param uniqueSuffix string = ''

// -----------------------------------------------------------------------------
// Variables - Model loading, filtering and unique suffix calculation
// -----------------------------------------------------------------------------

// Load models from YAML
var models = loadYamlContent('models.yaml')

// Calculate effective unique suffix
var effectiveUniqueSuffix = empty(uniqueSuffix) ? substring(uniqueString(resourceGroup().id), 0, 6) : uniqueSuffix

// Filter models based on modelFilterString
var effectiveModelFilter = empty(modelFilterString) ? [] : split(modelFilterString, ',')
var filteredModels = empty(effectiveModelFilter) ? models : filter(models, 
  m => contains(array(map(effectiveModelFilter, f => toLower(f))), toLower(m.name)))

// -----------------------------------------------------------------------------
// Resources
// -----------------------------------------------------------------------------

// Loop through each filtered model and deploy+update traffic using the new module
module model_deploy 'modelDeployWithTraffic.bicep' = [for model in filteredModels: {
  name: 'deploy-with-traffic-${model.name}'
  params: {
    location: location
    workspaceName: workspaceName
    modelName: model.name
    endpointName: ''
    deploymentName: ''
    modelId: model.deployment.modelId
    instanceType: model.deployment.instanceType
    instanceCount: model.deployment.instanceCount
    tags: tags
    requestSettings: model.deployment.requestSettings
    livenessProbe: model.deployment.livenessProbe
    environmentVariables: {}
    uniqueSuffix: effectiveUniqueSuffix
  }
}]

// -----------------------------------------------------------------------------
// Outputs
// -----------------------------------------------------------------------------
output endpoints array = [for (model, i) in filteredModels: {
  name: model.name
  env_name: model.env_name
  endpointName: model_deploy[i].outputs.endpointName
  id: model_deploy[i].outputs.endpointId
  scoringUri: model_deploy[i].outputs.scoringUri
  deploymentName: model_deploy[i].outputs.deploymentName
}]

output endpointSuffix string = effectiveUniqueSuffix


