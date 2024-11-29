# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

from azureml.core import Workspace
from openai import AzureOpenAI


def create_openai_client():
    """Plumbing to create the OpenAI client"""

    # Try to load endpoint URL and API key from the JSON file
    # (and load as environment variables)
    load_environment_variables("environment.json")

    # Try to get the key from environment
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")

    if api_key == "":
        # Try to get the key from AML workspace

        # Load the workspace
        ws = Workspace.from_config()

        # Access the linked key vault
        keyvault = ws.get_default_keyvault()

        # Get the secret
        api_key = keyvault.get_secret("azure-openai-api-key-westus")

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2024-02-01",
    )
    return client


def create_oai_assistant(client):
    """Creates assistant to keep track of prior responses"""
    # Assistant API example: https://github.com/openai/openai-python/blob/main/examples/assistant.py
    # Available in limited regions
    deployment = "gpt-4o"
    assistant = client.beta.assistants.create(
        name="Math Tutor",
        instructions="You are a categorizer. For each question answered, extract entities related to people's names and "
        " jobs and categorize them. You always return result in JSON. You reuse categories from past responses when possible",
        model=deployment,
        tools=[{"type": "code_interpreter"}],
    )
    return assistant.id
