// AI Services module for deploying Azure Cognitive Services with GPT models
// This module creates an AI Services account that can host GPT models

targetScope = 'resourceGroup'

@description('Azure region for deployment')
param location string

@description('Name of the AI Services account')
param aiServicesName string

@description('Tags to apply to all resources')
param tags object = {}

@description('List of principals to grant Cognitive Services OpenAI User role')
param grantAccessTo array = []

@description('Additional managed identities to grant access to')
param additionalIdentities array = []

@description('SKU for the AI Services account')
@allowed(['S0', 'F0'])
param sku string = 'S0'

@description('Whether to disable local authentication')
param disableLocalAuth bool = false

@description('Public network access setting')
@allowed(['Enabled', 'Disabled'])
param publicNetworkAccess string = 'Enabled'

// Create the AI Services account
resource aiServices 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: aiServicesName
  location: location
  tags: tags
  kind: 'AIServices'
  sku: {
    name: sku
  }
  properties: {
    apiProperties: {}
    customSubDomainName: toLower(aiServicesName)
    disableLocalAuth: disableLocalAuth
    publicNetworkAccess: publicNetworkAccess
    networkAcls: {
      defaultAction: 'Allow'
    }
  }
  identity: {
    type: 'SystemAssigned'
  }
}

// Role definition for Cognitive Services OpenAI User
var cognitiveServicesOpenAIUserRole = 'a97b65f3-24c7-4388-baec-2e87135dc908'

// Grant access to specified principals (only if local auth is disabled)
resource roleAssignments 'Microsoft.Authorization/roleAssignments@2022-04-01' = [
  for principal in grantAccessTo: if (!empty(principal.id) && disableLocalAuth) {
    name: guid(aiServices.id, principal.id, cognitiveServicesOpenAIUserRole)
    scope: aiServices
    properties: {
      roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', cognitiveServicesOpenAIUserRole)
      principalId: principal.id
      principalType: principal.type
    }
  }
]

// Grant access to additional managed identities (only if local auth is disabled)
resource additionalRoleAssignments 'Microsoft.Authorization/roleAssignments@2022-04-01' = [for (identity, index) in additionalIdentities: if (disableLocalAuth) {
  name: guid(aiServices.id, identity, cognitiveServicesOpenAIUserRole, 'additional')
  scope: aiServices
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', cognitiveServicesOpenAIUserRole)
    principalId: identity
    principalType: 'ServicePrincipal'
  }
}]

// Outputs
output aiServicesId string = aiServices.id
output aiServicesName string = aiServices.name
output aiServicesEndpoint string = aiServices.properties.endpoint
output aiServicesPrincipalId string = aiServices.identity.principalId
