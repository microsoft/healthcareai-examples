// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

@description('Azure region for deployment')
param location string

@description('Name of the Key Vault')
param keyVaultName string

@description('Tags to apply to all resources')
param tags object = {}

@description('List of principals to grant access to')
param grantAccessTo array

@description('Additional managed identities to assign access to')
param additionalIdentities array = []

var access = [for i in range(0, length(additionalIdentities)): {
  id: additionalIdentities[i]
  type: 'ServicePrincipal'
}]

var grantAccessToUpdated = concat(grantAccessTo, access)

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  tags: tags
  properties: {
    createMode: 'default'
    enabledForDeployment: false
    enabledForDiskEncryption: false
    enabledForTemplateDeployment: false
    enableSoftDelete: true
    enableRbacAuthorization: true
    publicNetworkAccess: 'Enabled'
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: 'Allow' 
    }
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
  }
}

resource secretsOfficer 'Microsoft.Authorization/roleDefinitions@2022-04-01' existing = {
  name: 'b86a8fe4-44ce-4948-aee5-eccb2c155cd7'
}

resource secretsOfficerAccess 'Microsoft.Authorization/roleAssignments@2022-04-01' = [
  for principal in grantAccessToUpdated: if (!empty(principal.id)) {
    name: guid(principal.id, keyVault.id, secretsOfficer.id)
    scope: keyVault
    properties: {
      roleDefinitionId: secretsOfficer.id
      principalId: principal.id
      principalType: principal.type
    }
  }
]


output keyVaultID string = keyVault.id
output keyVaultName string = keyVault.name
output keyVaultEndpoint string = keyVault.properties.vaultUri
