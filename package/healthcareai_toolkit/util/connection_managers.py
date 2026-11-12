# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Platform connection setup, independent of model variant and payload schema."""

import re
from abc import ABC, abstractmethod
from urllib.parse import urlsplit

from azure.ai.ml import MLClient

AML_ENDPOINT_ID = re.compile(
    r"/subscriptions/([^/\s?#]+)/resourceGroups/([^/\s?#]+)/providers/"
    r"Microsoft\.MachineLearningServices/workspaces/([^/\s?#]+)/onlineEndpoints/([^/\s?#]+)",
    re.IGNORECASE,
)
AML_ENDPOINT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


class ConnectionManager(ABC):
    """Own an inference target and API-key authorization headers."""

    @classmethod
    def from_endpoint(
        cls,
        endpoint,
        *,
        credential=None,
        api_key=None,
        api_path=None,
        exclude_managed_identity_credential=False,
    ):
        """Construct the platform connection manager for an endpoint value."""
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError("Set an endpoint ID, workspace-local name, or HTTPS URL.")
        if AML_ENDPOINT_ID.fullmatch(endpoint) or AML_ENDPOINT_NAME.fullmatch(endpoint):
            manager_class = AzureMLConnectionManager
        else:
            try:
                url = urlsplit(endpoint)
                hostname = url.hostname or ""
            except ValueError:
                raise ValueError(
                    "Malformed endpoint; supply an HTTPS inference URL."
                ) from None
            if url.scheme != "https" or url.username is not None:
                raise ValueError("Use an HTTPS inference URL without credentials.")
            if hostname.endswith(".services.ai.azure.com"):
                manager_class = FoundryConnectionManager
            elif hostname.endswith(".inference.ml.azure.com"):
                manager_class = AzureMLConnectionManager
            else:
                raise ValueError(
                    "Use an AML online endpoint ID/name or a .inference.ml.azure.com or .services.ai.azure.com URL."
                )
        return manager_class(
            endpoint,
            credential=credential,
            api_key=api_key,
            api_path=api_path,
            exclude_managed_identity_credential=exclude_managed_identity_credential,
        )

    @property
    @abstractmethod
    def target(self):
        raise NotImplementedError

    @property
    def headers(self):
        return {"Authorization": f"Bearer {self.api_key}"}

    def _set_api_key(self, api_key):
        if (
            not isinstance(api_key, str)
            or not api_key
            or not api_key.isascii()
            or any(not 33 <= ord(char) <= 126 for char in api_key)
        ):
            raise ValueError(
                "Set a nonempty ASCII API key without whitespace or control characters."
            )
        self.api_key = api_key


class AzureMLConnectionManager(ConnectionManager):
    """Connect by AML ID/name with management lookup, or directly by URL and key.

    Management lookup uses an explicit credential or DefaultAzureCredential,
    including inside Azure ML jobs. Automatic run-token authentication and
    private SDK management-URL overrides are not supported.
    """

    def __init__(
        self,
        resource_id,
        credential=None,
        *,
        api_key=None,
        api_path=None,
        exclude_managed_identity_credential=False,
    ):
        if isinstance(resource_id, str) and "://" in resource_id:
            self._target = resource_id
            self._set_api_key(api_key)
            return
        if not isinstance(resource_id, str) or not (
            AML_ENDPOINT_ID.fullmatch(resource_id)
            or AML_ENDPOINT_NAME.fullmatch(resource_id)
        ):
            raise ValueError(
                "Supply an AML online endpoint ID, workspace-local name, or URL."
            )
        if credential is None:
            from azure.identity import DefaultAzureCredential

            credential = DefaultAzureCredential(
                exclude_managed_identity_credential=exclude_managed_identity_credential
            )
        self._credential = credential
        self.initialize_client(resource_id)
        if api_key is None:
            api_key = self.ml_client.online_endpoints.get_keys(
                name=self.endpoint_name
            ).primary_key
        self._set_api_key(api_key)

    @property
    def target(self):
        return self._target

    def initialize_client(self, resource_id):
        uri_match = AML_ENDPOINT_ID.fullmatch(resource_id)

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

        self.ml_client = ml_client
        self.endpoint_name = endpoint_name
        self.endpoint = ml_client.online_endpoints.get(name=endpoint_name)
        if self.endpoint.auth_mode != "key":
            raise ValueError("Only API-key scoring is supported.")
        self._target = self.endpoint.scoring_uri


class FoundryConnectionManager(ConnectionManager):
    def __init__(
        self,
        endpoint,
        credential=None,
        *,
        api_key=None,
        api_path=None,
        exclude_managed_identity_credential=False,
    ):
        self._target = endpoint
        url = urlsplit(endpoint)
        if url.path in ("", "/"):
            if not api_path:
                raise ValueError("Supply the full inference URL for this model.")
            self._target = url._replace(path=f"/{api_path.lstrip('/')}").geturl()
        self._set_api_key(api_key)

    @property
    def target(self):
        return self._target
