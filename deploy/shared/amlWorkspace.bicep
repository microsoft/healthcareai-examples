targetScope = 'resourceGroup'

// -----------------------------------------------------------------------------
// Required parameters
// -----------------------------------------------------------------------------
@description('Azure region for the workspace')
param location string

@description('Name of the workspace')
param workspaceName string

@description('Tags to apply to resources')
param tags object = {}

@description('Storage account name')
param storageAccountName string

@description('Key vault name')
param keyVaultName string

@description('Container registry name')
param containerRegistryName string

@description('List of principals to grant access to the workspace')
param grantAccessTo array = []

@description('Additional managed identities to assign to the workspace')
param additionalIdentities array = []

@description('Whether to allow shared key access for the storage account and use identity-based access')
param allowSharedKeyAccess bool = true

@sys.description('Optional. The authentication mode used by the workspace when connecting to the default storage account.')
@allowed([
  'AccessKey'
  'Identity'
])
param systemDatastoresAuthMode string = allowSharedKeyAccess ? 'AccessKey' : 'Identity'

// Combine grantAccessTo with additionalIdentities
var access = [for i in range(0, length(additionalIdentities)): {
  id: additionalIdentities[i]
  type: 'ServicePrincipal'
}]

// Add workspace identity to the access list after workspace is created (using dependsOn in role assignments)
var grantAccessToUpdated = concat(grantAccessTo, access)

// -----------------------------------------------------------------------------
// Resources
// -----------------------------------------------------------------------------

// Create storage account
module storageAccount 'storageAccount.bicep' = {
  name: 'storage-account'
  params: {
    storageAccountName: storageAccountName
    location: location
    allowSharedKeyAccess: allowSharedKeyAccess
    tags: tags
    grantAccessTo: grantAccessToUpdated
    additionalIdentities: []
  }
}

// Create key vault
module keyVault 'keyVault.bicep' = {
  name: 'key-vault'
  params: {
    keyVaultName: keyVaultName
    location: location
    tags: tags
    grantAccessTo: grantAccessToUpdated
    additionalIdentities: []
  }
}

// Create container registry
module containerRegistry 'containerRegistry.bicep' = {
  name: 'container-registry'
  params: {
    containerRegistryName: containerRegistryName
    location: location
    tags: tags
    grantAccessTo: grantAccessToUpdated
    additionalIdentities: []
  }
}

// Create an Application Insights instance
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'ai-${uniqueString(resourceGroup().id)}'
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    DisableIpMasking: false
    DisableLocalAuth: false
    Flow_Type: 'Redfield'
    ForceCustomerStorageForProfiler: false
    ImmediatePurgeDataOn30Days: false
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
    Request_Source: 'rest'
  }
}

// Create Azure Machine Learning workspace
resource workspace 'Microsoft.MachineLearningServices/workspaces@2024-10-01-preview' = {
  name: workspaceName
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    friendlyName: workspaceName
    storageAccount: storageAccount.outputs.storageAccountID
    keyVault: keyVault.outputs.keyVaultID
    applicationInsights: appInsights.id
    containerRegistry: containerRegistry.outputs.containerRegistryID
    publicNetworkAccess: 'Enabled'
    v1LegacyMode: false
    systemDatastoresAuthMode: systemDatastoresAuthMode
  }
}

// -----------------------------------------------------------------------------
// Role assignments - correct separation of concerns:
// 1. Infrastructure modules handle external principal access to infrastructure resources
// 2. Workspace handles its system-assigned identity access to infrastructure resources  
// 3. Workspace handles external principal access to workspace itself
// -----------------------------------------------------------------------------

// Grant Key Vault Secrets User role to workspace system-assigned identity
resource keyVaultSecretsUser 'Microsoft.Authorization/roleDefinitions@2022-04-01' existing = {
  name: '4633458b-17de-408a-b874-0445c86b69e6'
}

resource keyVaultRef 'Microsoft.KeyVault/vaults@2023-07-01' existing = {
  name: keyVaultName
}

resource workspaceKeyVaultRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVaultName, workspace.id, keyVaultSecretsUser.id)
  scope: keyVaultRef
  properties: {
    roleDefinitionId: keyVaultSecretsUser.id
    principalId: workspace.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// Grant AcrPull role to workspace system-assigned identity
resource acrPull 'Microsoft.Authorization/roleDefinitions@2022-04-01' existing = {
  name: '7f951dda-4ed3-4680-a7ca-43fe172d538d'
}

resource containerRegistryRef 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: containerRegistryName
}

resource workspaceContainerRegistryRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(containerRegistryName, workspace.id, acrPull.id)
  scope: containerRegistryRef
  properties: {
    roleDefinitionId: acrPull.id
    principalId: workspace.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// Grant Managed Identity Operator role to workspace system-assigned identity (scoped to resource group)
resource managedIdentityOperator 'Microsoft.Authorization/roleDefinitions@2022-04-01' existing = {
  name: 'f1a07417-d97a-45cb-824c-7a7467783830'
}

resource managedIdentityOperatorRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(resourceGroup().id, workspace.id, managedIdentityOperator.id)
  scope: resourceGroup()
  properties: {
    roleDefinitionId: managedIdentityOperator.id
    principalId: workspace.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// Grant AzureML Data Scientist role to external principals for workspace access
resource azureMLDataScientist 'Microsoft.Authorization/roleDefinitions@2022-04-01' existing = {
  name: 'f6c7c914-8db3-469d-8ca1-694a8f32e121'
}

resource workspaceAccessRoleAssignments 'Microsoft.Authorization/roleAssignments@2022-04-01' = [
  for principal in grantAccessToUpdated: if (!empty(principal.id)) {
    name: guid(workspace.id, principal.id, azureMLDataScientist.id)
    scope: workspace
    properties: {
      roleDefinitionId: azureMLDataScientist.id
      principalId: principal.id
      principalType: principal.type
    }
  }
]



// -----------------------------------------------------------------------------
// Outputs
// -----------------------------------------------------------------------------
output workspaceId string = workspace.id
output workspaceName string = workspace.name
output workspaceLocation string = workspace.location

// Application Insights outputs
output appInsightsInstrumentationKey string = appInsights.properties.InstrumentationKey
output appInsightsId string = appInsights.id

// Storage account outputs
output storageAccountId string = storageAccount.outputs.storageAccountID
output storageAccountName string = storageAccount.outputs.storageAccountName
output storageAccountBlobEndpoint string = storageAccount.outputs.storageAccountBlobEndpoint

// Key vault outputs
output keyVaultId string = keyVault.outputs.keyVaultID
output keyVaultName string = keyVault.outputs.keyVaultName
output keyVaultEndpoint string = keyVault.outputs.keyVaultEndpoint

// Container registry outputs
output containerRegistryId string = containerRegistry.outputs.containerRegistryID
output containerRegistryName string = containerRegistry.outputs.containerRegistryName
output containerRegistryLoginServer string = containerRegistry.outputs.containerRegistryLoginServer
