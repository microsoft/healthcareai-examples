# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from openai import AzureOpenAI
from healthcareai_toolkit import settings


def create_openai_client():
    """Plumbing to create the OpenAI client"""
    endpoint = settings.AZURE_OPENAI_ENDPOINT
    api_key = settings.AZURE_OPENAI_API_KEY

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2024-02-01",
    )
    return client
