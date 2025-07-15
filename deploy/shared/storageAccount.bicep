// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

@description('Specifies the name of the Azure Storage account.')
param storageAccountName string

@description('Specifies the location in which the Azure Storage resources should be deployed.')
param location string

@description('Allow shared key access to the storage account (default: true for compatibility)')
param allowSharedKeyAccess bool = true

@description('Tags to apply to all resources')
param tags object = {}

@description('List of principals to grant access to')
param grantAccessTo array = []

@description('Additional managed identities to assign access to')
param additionalIdentities array = []

var access = [for i in range(0, length(additionalIdentities)): {
  id: additionalIdentities[i]
  type: 'ServicePrincipal'
}]

var grantAccessToUpdated = concat(grantAccessTo, access)

resource sa 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountName
  location: location
  tags: tags
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    allowSharedKeyAccess: allowSharedKeyAccess
    accessTier: 'Hot'
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
  }
}

resource blobServices 'Microsoft.Storage/storageAccounts/blobServices@2023-01-01' = {
  parent: sa
  name: 'default'
}

// Storage Blob Data Owner
resource storageBlobDataOwner 'Microsoft.Authorization/roleDefinitions@2022-04-01' existing = {
  name: 'b7e6dc6d-f1e8-4753-8033-0f276bb0955b'
}

resource blobDataAccess 'Microsoft.Authorization/roleAssignments@2022-04-01' = [
  for principal in grantAccessToUpdated: if (!empty(principal.id)) {
    name: guid(principal.id, sa.id, storageBlobDataOwner.id)
    scope: sa
    properties: {
      roleDefinitionId: storageBlobDataOwner.id
      principalId: principal.id
      principalType: principal.type
    }
  }
]

// Storage Table Data Contributor
resource storageTableDataContributor 'Microsoft.Authorization/roleDefinitions@2022-04-01' existing = {
  name: '0a9a7e1f-b9d0-4cc4-a60d-0319b160aaa3'
}

resource tableDataAccess 'Microsoft.Authorization/roleAssignments@2022-04-01' = [
  for principal in grantAccessToUpdated: if (!empty(principal.id)) {
    name: guid(principal.id, sa.id, storageTableDataContributor.id)
    scope: sa
    properties: {
      roleDefinitionId: storageTableDataContributor.id
      principalId: principal.id
      principalType: principal.type
    }
  }
]

// Storage Queue Data Contributor
resource storageQueueDataContributor 'Microsoft.Authorization/roleDefinitions@2022-04-01' existing = {
  name: '974c5e8b-45b9-4653-ba55-5f855dd0fb88'
}

resource queueDataAccess 'Microsoft.Authorization/roleAssignments@2022-04-01' = [
  for principal in grantAccessToUpdated: if (!empty(principal.id)) {
    name: guid(principal.id, sa.id, storageQueueDataContributor.id)
    scope: sa
    properties: {
      roleDefinitionId: storageQueueDataContributor.id
      principalId: principal.id
      principalType: principal.type
    }
  }
]

// Storage File Data SMB Share Elevated Contributor
resource storageFileDataSMBShareElevatedContributor 'Microsoft.Authorization/roleDefinitions@2022-04-01' existing = {
  name: 'a7264617-510b-434b-a828-9731dc254ea7'
}

resource fileShareAccess 'Microsoft.Authorization/roleAssignments@2022-04-01' = [
  for principal in grantAccessToUpdated: if (!empty(principal.id)) {
    name: guid(principal.id, sa.id, storageFileDataSMBShareElevatedContributor.id)
    scope: sa
    properties: {
      roleDefinitionId: storageFileDataSMBShareElevatedContributor.id
      principalId: principal.id
      principalType: principal.type
    }
  }
]

// Storage File Data Privileged Contributor
resource storageFileDataPrivilegedContributor 'Microsoft.Authorization/roleDefinitions@2022-04-01' existing = {
  name: '69566ab7-960f-475b-8e7c-b3118f30c6bd'
}

resource fileDataPrivilegedAccess 'Microsoft.Authorization/roleAssignments@2022-04-01' = [
  for principal in grantAccessToUpdated: if (!empty(principal.id)) {
    name: guid(principal.id, sa.id, storageFileDataPrivilegedContributor.id)
    scope: sa
    properties: {
      roleDefinitionId: storageFileDataPrivilegedContributor.id
      principalId: principal.id
      principalType: principal.type
    }
  }
]

// Reader and Data Access
resource readerAndDataAccess 'Microsoft.Authorization/roleDefinitions@2022-04-01' existing = {
  name: 'c12c1c16-33a1-487b-954d-41c89c60f349'
}

resource keyAccess 'Microsoft.Authorization/roleAssignments@2022-04-01' = [
  for principal in grantAccessToUpdated: if (!empty(principal.id)) {
    name: guid(principal.id, sa.id, readerAndDataAccess.id)
    scope: sa
    properties: {
      roleDefinitionId: readerAndDataAccess.id
      principalId: principal.id
      principalType: principal.type
    }
  }
]

output storageAccountID string = sa.id
output storageAccountName string = sa.name
output storageAccountBlobEndpoint string = sa.properties.primaryEndpoints.blob
