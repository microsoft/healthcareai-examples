#!/usr/bin/env python3
"""
Simple cleanup script for Azure resources created by azd deployments.

This handles the limitation that 'azd down' doesn't work for existing workspace deployments.
"""

import argparse
import asyncio
import sys
from collections import defaultdict
from functools import wraps
from itertools import groupby
from typing import Dict, List, Set, Tuple

from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.ai.ml.entities import OnlineEndpoint

try:
    from azure.ai.ml import MLClient
except (ImportError, ModuleNotFoundError) as e:
    print(
        "Error: azure.ai.ml (AzureML SDK v2) is not installed.\n"
        "If you are on an AzureML VM, the 'azureml_py310_sdkv2' environment is recommended.\n"
        "If not, you can install the SDK with: pip install azure-ai-ml"
    )
    raise e

try:
    from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
except (ImportError, ModuleNotFoundError) as e:
    print(
        "Warning: Azure Cognitive Services management SDK not installed.\n"
        "Some purge operations may not be available.\n"
        "Install with: pip install azure-mgmt-cognitiveservices"
    )

from utils import (
    load_azd_env_vars,
    detect_deployment_type,
    GREEN,
    YELLOW,
    RED,
    BLUE,
    CYAN,
    BOLD,
    END,
)


def async_wrap(func):
    """Decorator to automatically wrap blocking functions with asyncio.to_thread."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper


def get_deletion_order(resource: Dict) -> int:
    """Return the deletion order for a resource. Lower numbers are deleted first."""
    resource_type = resource.type.lower()

    # Order 0: ML deployments (must be deleted before endpoints)
    if "onlineendpoints/deployments" in resource_type:
        return 0

    # Order 1: ML endpoints (must be deleted before workspaces)
    if "onlineendpoints" in resource_type:
        return 1

    # Order 2: Everything else (workspaces, storage, etc.)
    return 2


async def delete_resources_in_order(
    resource_client: ResourceManagementClient,
    resources: List[Dict],
    ml_client: MLClient = None,
    purge: bool = False,
    credential=None,
    subscription_id: str = None,
    resource_group: str = None,
):
    """Delete resources in dependency order."""
    if not resources:
        return

    # Sort resources by deletion order, then group by order
    sorted_resources = sorted(resources, key=get_deletion_order)

    print(f"\nDeleting {len(resources)} resources in dependency order...")

    total_successful = 0
    total_failed = 0

    # Group resources by deletion order using groupby
    for order, group in groupby(sorted_resources, key=get_deletion_order):
        group_resources = list(group)
        print(f"\nGroup {order}:")

        # Run all resources in this order group concurrently
        deletion_tasks = [
            delete_resource_async(
                resource_client,
                resource,
                ml_client,
                purge,
                credential,
                subscription_id,
                resource_group,
            )
            for resource in group_resources
        ]

        # Wait for all deletions in this group to complete
        results = await asyncio.gather(*deletion_tasks, return_exceptions=True)

        # Count successes and failures
        group_successful = sum(1 for result in results if result is True)
        group_failed = len(results) - group_successful

        total_successful += group_successful
        total_failed += group_failed

        print(
            f"  Group {order} completed: {len(group_resources)} resources ({group_successful} successful, {group_failed} failed)"
        )

    print(
        f"\nOverall deletion summary: {total_successful} successful, {total_failed} failed"
    )


def find_azd_resources(
    resource_client: ResourceManagementClient, resource_group: str, env_name: str
) -> List[Dict]:
    """Find all resources tagged with azd-env-name in the specific resource group."""
    try:
        resources = list(
            resource_client.resources.list_by_resource_group(
                resource_group,
                filter=f"tagName eq 'azd-env-name' and tagValue eq '{env_name}'",
            )
        )
        return resources
    except Exception as e:
        print(f"Error finding resources in resource group '{resource_group}': {e}")
        return []


def group_resources_by_relationships(azd_resources: List[Dict]) -> Dict:
    """Group resources by their logical relationships and types using resource ID tree structure."""
    # Use defaultdict to automatically create lists for each resource type
    resources = defaultdict(list)

    # Create mapping from endpoint ID to its deployments
    endpoint_id_to_deployments = {}

    for resource in azd_resources:
        resource_type = resource.type.lower()

        if "onlineendpoints/deployments" in resource_type:
            resources["ml_deployments"].append(resource)
            # Extract parent endpoint ID from deployment resource ID
            # Deployment ID: /subscriptions/.../workspaces/ws/onlineEndpoints/ep/deployments/dep
            # Parent endpoint ID: /subscriptions/.../workspaces/ws/onlineEndpoints/ep
            endpoint_id = resource.id.rsplit("/deployments/", 1)[0]

            if endpoint_id not in endpoint_id_to_deployments:
                endpoint_id_to_deployments[endpoint_id] = []
            endpoint_id_to_deployments[endpoint_id].append(resource)

        elif "onlineendpoints" in resource_type:
            resources["ml_endpoints"].append(resource)
        elif (
            "microsoft.machinelearningservices/workspaces" in resource_type
            or "microsoft.cognitiveservices" in resource_type
        ):
            resources["ai_services"].append(resource)

        else:
            resources["infrastructure"].append(resource)

    return resources, endpoint_id_to_deployments


def print_resources(azd_resources: List[Dict], delete_all: bool = False) -> List[Dict]:
    """Print summary of resources organized by relationships, showing what will be deleted.

    Returns the list of resources that will actually be deleted.
    """
    print(f"\n{BOLD}{'=' * 60}{END}")
    print(f"{BOLD}{BLUE}Discovered Resources{END}")
    print(f"{BOLD}{'=' * 60}{END}")

    if not azd_resources:
        print(f"{YELLOW}No azd-tagged resources found.{END}")
        return []

    total_infrastructure = len(azd_resources)
    grouped, deps = group_resources_by_relationships(azd_resources)

    # ML Endpoints & Deployments section
    if grouped["ml_endpoints"] or grouped["ml_deployments"]:
        print(f"\n{CYAN}ML Endpoints & Deployments:{END}")

        # Show endpoints with their deployments using ID-based mapping
        for endpoint in grouped["ml_endpoints"]:
            endpoint_id = endpoint.id
            deployments = deps.get(endpoint_id, [])

            endpoint_delete_marker = f" {RED}✓ WILL DELETE{END}" if delete_all else ""
            print(
                f"  └── {BLUE}{endpoint.name}{END} ({endpoint.type}){endpoint_delete_marker}"
            )
            for deployment in deployments:
                print(
                    f"      ├── {deployment.name} ({deployment.type}) {RED}✓ WILL DELETE{END}"
                )

        # Show any orphaned deployments (endpoints not found)
        all_endpoint_ids = {ep.id for ep in grouped["ml_endpoints"]}
        for endpoint_id, deployments in deps.items():
            if endpoint_id not in all_endpoint_ids:
                # Extract endpoint name from ID for display
                endpoint_name = endpoint_id.split("/")[-1]
                print(f"  └── {YELLOW}{endpoint_name} (endpoint not found){END}")
                for deployment in deployments:
                    print(
                        f"      ├── {deployment.name} ({deployment.type}) {RED}✓ WILL DELETE{END}"
                    )

    # AI Workspaces section
    if grouped["ai_services"]:
        print(f"\n{CYAN}AI Services:{END}")
        for workspace in grouped["ai_services"]:
            delete_marker = f" {RED}✓ WILL DELETE{END}" if delete_all else ""
            print(
                f"  ├── {BLUE}{workspace.name}{END} ({workspace.type}){delete_marker}"
            )

    # Infrastructure section
    if grouped["infrastructure"]:
        print(f"\n{CYAN}Infrastructure:{END}")
        for resource in grouped["infrastructure"]:
            delete_marker = f" {RED}✓ WILL DELETE{END}" if delete_all else ""
            print(f"  ├── {BLUE}{resource.name}{END} ({resource.type}){delete_marker}")

    # Build list of resources that will actually be deleted
    resources_to_delete = []

    # Always delete deployments
    resources_to_delete.extend(grouped["ml_deployments"])

    # Delete other resources only if --all flag is used
    if delete_all:
        resources_to_delete.extend(grouped["ml_endpoints"])
        resources_to_delete.extend(grouped["ai_services"])
        resources_to_delete.extend(grouped["infrastructure"])

    # Deletion summary
    print(f"\n{BOLD}{'=' * 60}{END}")
    print(f"\n{BOLD}Deletion Summary:{END}")
    print(f"{BOLD}{'=' * 60}{END}")
    if resources_to_delete:
        print(f"{RED}✓ Resources to DELETE ({len(resources_to_delete)}):{END}")
        for resource in resources_to_delete:  # Show first 5
            print(f"  - {resource.name}")
    else:
        print(f"{GREEN}✓ No resources will be deleted{END}")

    if not delete_all and (
        grouped["ml_endpoints"] or grouped["ai_services"] or grouped["infrastructure"]
    ):
        print(
            f"\n{YELLOW}Note: Use --all flag to delete infrastructure resources ({total_infrastructure} resources){END}"
        )

    print(f"{BOLD}{'=' * 60}{END}")

    return resources_to_delete


async def delete_resource_async(
    resource_client: ResourceManagementClient,
    resource: Dict,
    ml_client: MLClient = None,
    purge: bool = False,
    credential=None,
    subscription_id: str = None,
    resource_group: str = None,
):
    """Delete a single resource asynchronously."""
    try:
        print(f"  Deleting {resource.name} ({resource.type})...")

        if "onlineendpoints/deployments" in resource.type.lower() and ml_client:
            await delete_ml_deployment(ml_client, resource.id)
        elif "onlineendpoints" in resource.type.lower() and ml_client:
            await delete_ml_endpoint(ml_client, resource.id)
        elif "microsoft.cognitiveservices/accounts" in resource.type.lower():
            await delete_cognitive_services(
                resource_client,
                resource,
                credential,
                subscription_id,
                resource_group,
                purge,
            )
        else:
            print(f"  Warning: Unsupported resource type {resource.type} - skipping")
            return False

        print(f"  ✓ Deleted {resource.name}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to delete {resource.name}: {e}")
        return False


def remove_deployment_traffic(
    ml_client: MLClient, endpoint_name: str, deployment_name: str
):
    """Set deployment traffic to 0 before deletion."""
    try:
        # Get endpoint, set traffic to 0, update endpoint
        endpoint = ml_client.online_endpoints.get(endpoint_name)
        endpoint.traffic[deployment_name] = 0
        endpoint.identity.principal_id = None
        endpoint.identity.tenant_id = None
        ml_client.online_endpoints.begin_create_or_update(endpoint).wait()
        print(f"      ✓ Traffic set to 0 for {deployment_name}")

    except Exception as e:
        print(f"      Warning: Could not remove traffic from {deployment_name}: {e}")
        print(f"      Attempting deletion anyway...")


@async_wrap
def delete_ml_deployment(ml_client: MLClient, resource_id: str):
    """Delete an Azure ML endpoint deployment."""
    # Parse endpoint and deployment names from resource path
    path_parts = resource_id.split("/")
    endpoint_name = path_parts[path_parts.index("onlineEndpoints") + 1]
    deployment_name = path_parts[path_parts.index("deployments") + 1]

    # Remove traffic first, then delete
    remove_deployment_traffic(ml_client, endpoint_name, deployment_name)
    ml_client.online_deployments.begin_delete(
        name=deployment_name, endpoint_name=endpoint_name
    ).wait()


@async_wrap
def delete_ml_endpoint(ml_client: MLClient, resource_id: str):
    """Delete an Azure ML online endpoint."""
    # Parse endpoint name from resource path
    path_parts = resource_id.split("/")
    endpoint_name = path_parts[path_parts.index("onlineEndpoints") + 1]
    ml_client.online_endpoints.begin_delete(name=endpoint_name).wait()


def delete_resource_by_id(
    resource_client: ResourceManagementClient, resource_id: str, api_version: str
):
    """Delete resource by ID - wrapped to be async."""
    return resource_client.resources.begin_delete_by_id(
        resource_id, api_version=api_version
    ).wait()


@async_wrap
def delete_cognitive_services(
    resource_client: ResourceManagementClient,
    resource: Dict,
    credential,
    subscription_id: str,
    resource_group: str,
    purge: bool = False,
):
    """Delete a Cognitive Services account and optionally purge it."""
    api_version = "2024-10-01"  # Latest stable API version for Cognitive Services
    delete_resource_by_id(resource_client, resource.id, api_version)

    if purge:
        try:
            from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient

            cs_client = CognitiveServicesManagementClient(credential, subscription_id)
            # Extract account name and location from resource
            account_name = resource.name
            location = resource.location
            print(f"    Purging Cognitive Services account: {account_name}")
            cs_client.deleted_accounts.begin_purge(
                location, resource_group, account_name
            ).wait()
            print(f"    ✓ Purged Cognitive Services account: {account_name}")
        except Exception as e:
            print(
                f"    Warning: Could not purge Cognitive Services account {resource.name}: {e}"
            )


def main():
    parser = argparse.ArgumentParser(description="Clean up azd deployment resources")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Delete all azd-tagged resources (default: only model deployments)",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Purge Cognitive Services immediately after deletion",
    )
    args = parser.parse_args()

    # Prevent deleting all resources for 'fresh' deployments
    if args.all:
        deployment_type = detect_deployment_type()
        if deployment_type == "fresh":
            print(
                "Error: For fresh deployments, use the 'azd down' command instead of this cleanup script."
            )
            sys.exit(1)

    return asyncio.run(async_main(args))


async def async_main(args):
    try:
        # Detect deployment type
        deployment_type = detect_deployment_type()

        # Load azd environment
        print("Loading azd environment...")
        config = load_azd_env_vars()

        env_name = config.get("AZURE_ENV_NAME")
        subscription_id = config.get("AZURE_SUBSCRIPTION_ID")
        resource_group = config.get("AZURE_RESOURCE_GROUP")
        workspace_name = config.get("AZUREML_WORKSPACE_NAME")
        location = config.get("AZURE_LOCATION", "eastus")  # Default fallback

        if not all([env_name, subscription_id, resource_group]):
            print("Error: Missing required azd environment variables")
            print(
                "Required: AZURE_ENV_NAME, AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP"
            )
            return 1

        print(f"Environment: {env_name}")
        print(f"Subscription: {subscription_id}")
        print(f"Resource Group: {resource_group}")
        print(f"Location: {location}")
        print(f"Deployment Type: {deployment_type}")

        if not args.all:
            print("\nMode: Delete model deployments only (they charge per hour)")
            print("Use --all flag to also delete infrastructure resources")
        else:
            print("\nMode: Delete ALL azd-tagged resources")

        if args.purge:
            print("Purge mode: Will also purge soft-deleted resources")

        # Initialize clients
        credential = DefaultAzureCredential()
        resource_client = ResourceManagementClient(credential, subscription_id)

        # Initialize ML client if workspace exists
        ml_client = None
        if workspace_name:
            try:
                ml_client = MLClient(
                    credential=credential,
                    subscription_id=subscription_id,
                    resource_group_name=resource_group,
                    workspace_name=workspace_name,
                )
                print(f"Initialized ML client for workspace: {workspace_name}")
            except Exception as e:
                print(f"Warning: Could not initialize ML client: {e}")
                print("Will fall back to Resource Management API for all resources")

        # Find all azd-tagged resources in the specific resource group
        print(
            f"\nScanning for resources tagged with azd-env-name='{env_name}' in '{resource_group}'..."
        )
        azd_resources = find_azd_resources(resource_client, resource_group, env_name)

        # Show what we found and what will be deleted
        resources_to_delete = print_resources(azd_resources, args.all)

        if not resources_to_delete:
            print("\nNo resources found to delete.")
            return 0

        # Confirm deletion
        if not args.yes:
            purge_text = " (with purging of Cognitive Services)" if args.purge else ""
            response = input(
                f"\nDelete {len(resources_to_delete)} resources{purge_text}? [y/N]: "
            )
            if response.lower() != "y":
                print("Cancelled.")
                return 0

        # Delete resources
        print(f"\nStarting deletion of {len(resources_to_delete)} resources...")

        await delete_resources_in_order(
            resource_client,
            resources_to_delete,
            ml_client,
            args.purge,
            credential,
            subscription_id,
            resource_group,
        )

        print("\n✓ Cleanup completed!")

        return 0

    except KeyboardInterrupt:
        print("\nCancelled by user.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
