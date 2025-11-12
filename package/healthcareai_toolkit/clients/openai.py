# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from openai import AzureOpenAI
from healthcareai_toolkit import settings


def create_openai_client():
    """Create Azure OpenAI client with configuration from settings."""
    client = AzureOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
    )
    return client
