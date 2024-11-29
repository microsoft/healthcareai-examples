# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import traceback
from abc import ABC, abstractmethod
import re

from azure.ai.ml import MLClient
from azure.ai.ml.entities import WorkspaceConnection


class WorkspaceManagerBase(ABC):
    def __init__(self, resource_id, credential=None) -> None:
        self._set_credential(credential)
        self.initialize_client(resource_id)

    @property
    def headers(self):
        return {}

    @property
    @abstractmethod
    def target(self):
        raise NotImplementedError("implement me!")

    @abstractmethod
    def initialize_client(self, resource_id):
        raise NotImplementedError("implement me!")

    def _set_credential(self, credential):
        if credential is None:
            if os.environ.get("AZUREML_RUN_ID", None) is not None:
                from azureml.dataprep.api._aml_auth._azureml_token_authentication import (
                    AzureMLTokenAuthentication,
                )

                self._credential = (
                    AzureMLTokenAuthentication._initialize_aml_token_auth()
                )
            else:
                from azure.identity import DefaultAzureCredential

                self._credential = DefaultAzureCredential()
        else:
            self._credential = credential

    def handle_exception(self, exception):
        tb = traceback.format_exc()
        raise Exception(
            f"Error encountered while attempting to authentication token: {tb}"
        ) from exception


class WorkspaceEndpointManager(WorkspaceManagerBase):
    @property
    def target(self):
        return self.endpoint.scoring_uri

    @property
    def headers(self):
        value = {"Authorization": f"Bearer {self.api_key}"}
        if self.deployment_name is not None:
            pass
        return value

    def initialize_client(self, resource_id):
        uri_match = re.match(
            r"/subscriptions/(.*)/resourceGroups/(.*)/providers/Microsoft.MachineLearningServices/workspaces/(.*)/onlineEndpoints/(.*)",  # noqa: E501
            resource_id,
            flags=re.IGNORECASE,
        )

        if uri_match is None:
            ml_client = MLClient.from_config(self._credential)
            endpoint_name = resource_id
        else:
            subscription_id = uri_match.group(1)
            resource_group_name = uri_match.group(2)
            workspace_name = uri_match.group(3)
            endpoint_name = uri_match.group(4)
            ml_client = MLClient(
                credential=self._credential,
                subscription_id=subscription_id,
                resource_group_name=resource_group_name,
                workspace_name=workspace_name,
            )
            if os.environ.get("AZUREML_RUN_ID", None) is not None:
                ml_client.online_endpoints._online_operation._client._base_url = f"{os.environ.get('AZUREML_SERVICE_ENDPOINT')}/rp/workspaces"  # noqa: E501
                ml_client.online_endpoints._online_deployment_operation._client._base_url = f"{os.environ.get('AZUREML_SERVICE_ENDPOINT')}/rp/workspaces"  # noqa: E501

            print(
                f"Using ml_client base_url 1: {ml_client.online_endpoints._online_operation._client._base_url}"
            )
            print(
                f"Using ml_client base_url 2: {ml_client.online_endpoints._online_deployment_operation._client._base_url}"
            )

            self.ml_client = ml_client
            self.endpoint = ml_client.online_endpoints.get(name=endpoint_name)

            if self.endpoint.auth_mode != "key":
                raise NotImplementedError("only api key is implemented")

            keys = self.ml_client.online_endpoints.get_keys(name=endpoint_name)
            self.api_key = keys.primary_key
            self.deployment_name = None


class WorkspaceConnectionManager(WorkspaceManagerBase):
    def __init__(
        self, connection, auth_header_key="Ocp-Apim-Subscription-Key", credential=None
    ) -> None:
        super().__init__(credential)

        try:
            self.initialize_client(connection, auth_header_key)
        except Exception as e:
            self.handle_exception(e)

        uri_match = re.match(
            r"/subscriptions/(.*)/resourceGroups/(.*)/providers/Microsoft.MachineLearningServices/workspaces/(.*)/connections/(.*)",  # noqa: E501
            connection,
            flags=re.IGNORECASE,
        )

        subscription_id = uri_match.group(1)
        resource_group_name = uri_match.group(2)
        workspace_name = uri_match.group(3)
        ml_client = MLClient(
            credential=self._credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group_name,
            workspace_name=workspace_name,
        )
        if os.environ.get("AZUREML_RUN_ID", None) is not None:
            ml_client.connections._operation._client._base_url = f"{os.environ.get('AZUREML_SERVICE_ENDPOINT')}/rp/workspaces"  # noqa: E501
        print(
            f"Using ml_client base_url: {ml_client.connections._operation._client._base_url}"
        )
        list_secrets_response = ml_client.connections._operation.list_secrets(
            connection_name=uri_match.group(4),
            resource_group_name=ml_client.resource_group_name,
            workspace_name=ml_client.workspace_name,
        )
        connection = WorkspaceConnection._from_rest_object(list_secrets_response)
        print(f"Retrieved Workspace Connection: {connection.id}")

        auth_header_key = (
            connection.tags.get("azureml.AuthHeaderKey", None) or auth_header_key
        )

        def is_header_key(k):
            return k.startswith("azureml.headers=")

        def get_header_key(k):
            return k.split("=")[1]

        extra_headers = {
            get_header_key(tag_key): tag_value
            for tag_key, tag_value in connection.tags.items()
            if is_header_key(tag_key)
        }

        print(connection.type)
        self.set_target_and_auth_headers(
            connection.target,
            {
                auth_header_key: connection.credentials.key,
                # **extra_headers
            },
        )
