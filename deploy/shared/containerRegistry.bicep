// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

@description('Specifies the name of the Azure Container Registry.')
param containerRegistryName string

@description('Specifies the location in which the Azure Container Registry should be deployed.')
param location string

param tags object = {}
param grantAccessTo array = []
param additionalIdentities array = []

var access = [for i in range(0, length(additionalIdentities)): {
  id: additionalIdentities[i]
  type: 'ServicePrincipal'
}]

var grantAccessToUpdated = concat(grantAccessTo, access)

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: containerRegistryName
  location: location
  tags: tags
  sku: {
    name: 'Standard'
  }
  properties: {
    adminUserEnabled: false
    publicNetworkAccess: 'Enabled'
    networkRuleBypassOptions: 'AzureServices'
    zoneRedundancy: 'Disabled'
  }
}

resource acrPull 'Microsoft.Authorization/roleDefinitions@2022-04-01' existing = {
  name: '7f951dda-4ed3-4680-a7ca-43fe172d538d' // AcrPull
}

resource acrPullAccess 'Microsoft.Authorization/roleAssignments@2022-04-01' = [
  for principal in grantAccessToUpdated: if (!empty(principal.id)) {
    name: guid(principal.id, acr.id, acrPull.id)
    scope: acr
    properties: {
      roleDefinitionId: acrPull.id
      principalId: principal.id
      principalType: principal.type
    }
  }
]

output containerRegistryID string = acr.id
output containerRegistryName string = acr.name
output containerRegistryLoginServer string = acr.properties.loginServer
