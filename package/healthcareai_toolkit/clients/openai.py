# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from urllib.parse import urlparse

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
from openai import AzureOpenAI

from healthcareai_toolkit import settings


class AzureOpenAIConfig:
    """Configuration for Azure OpenAI, supporting both key auth and Entra keyless auth."""

    def __init__(self, endpoint=None, deployment=None, api_version=None, api_key=None):
        self.endpoint = endpoint
        self.deployment = deployment
        self.api_version = api_version
        self._api_key = api_key
        self._key_lookup_done = False

    def _credential(self):
        return DefaultAzureCredential(
            exclude_managed_identity_credential=settings.DISABLE_MANAGED_IDENTITY
        )

    @property
    def api_key(self):
        if self._api_key is None and not self._key_lookup_done:
            self._key_lookup_done = True
            self._api_key = self._fetch_key()
        return self._api_key

    @property
    def use_key_auth(self):
        return self.api_key is not None

    @staticmethod
    def _mask_key(key):
        if not key:
            return None
        if len(key) > 16:
            return key[:8] + "*" * (len(key) - 16) + key[-8:]
        return "***HIDDEN***"

    @property
    def masked_key(self):
        """Masked form of the configured key, or None when no key is set."""
        return self._mask_key(self._api_key)

    def __repr__(self):
        auth = "key" if self._api_key else "keyless"
        return (
            f"AzureOpenAIConfig(endpoint={self.endpoint!r}, "
            f"deployment={self.deployment!r}, api_version={self.api_version!r}, auth={auth!r})"
        )

    def _fetch_key(self):
        sub = os.environ.get("AZURE_SUBSCRIPTION_ID")
        rg = os.environ.get("AZURE_RESOURCE_GROUP")
        if not sub or not rg or not self.endpoint:
            return None
        hostname = urlparse(self.endpoint).hostname
        if not hostname:
            return None
        account = hostname.split(".")[0]
        try:
            client = CognitiveServicesManagementClient(self._credential(), sub)
            acct = client.accounts.get(rg, account)
            if acct.properties.disable_local_auth:
                return None
            keys = client.accounts.list_keys(rg, account)
            return keys.key1
        except (
            Exception
        ):  # slopcop: ignore[no-broad-except] best-effort key fetch; fall back to Entra on any failure
            return None

    def create_openai_client(self) -> AzureOpenAI:
        """Create and return an openai.AzureOpenAI client."""
        if self.use_key_auth:
            return AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
            )
        token_provider = get_bearer_token_provider(
            self._credential(), "https://cognitiveservices.azure.com/.default"
        )
        return AzureOpenAI(
            azure_endpoint=self.endpoint,
            azure_ad_token_provider=token_provider,
            api_version=self.api_version,
        )

    def create_maf_chat_client(self):
        """Create and return an agent_framework OpenAIChatClient."""
        try:
            # reason: optional dep; agent-framework is only needed for the MAF client
            from agent_framework.openai import OpenAIChatClient
        except ImportError as e:
            raise NotImplementedError(
                "agent-framework-openai is not installed; install it to use create_maf_chat_client()"
            ) from e

        if self.use_key_auth:
            return OpenAIChatClient(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                model=self.deployment,
                api_version="preview",
            )
        return OpenAIChatClient(
            azure_endpoint=self.endpoint,
            credential=self._credential(),
            model=self.deployment,
            api_version="preview",
        )


defaultConfig = AzureOpenAIConfig(
    settings.AZURE_OPENAI_ENDPOINT,
    settings.AZURE_OPENAI_DEPLOYMENT_NAME,
    settings.AZURE_OPENAI_API_VERSION,
    settings.AZURE_OPENAI_API_KEY,
)
