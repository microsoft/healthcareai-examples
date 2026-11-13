# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Representative secret-safe failures with offline transport guards."""

import traceback

import pytest


@pytest.mark.parametrize("key", ["prefixAB-secret-middle-suffixYZ", "short", "", None])
def test_cli_displays_key_presence_only(toolkit, monkeypatch, capsys, key):
    from healthcareai_toolkit.cli import test_endpoints
    from healthcareai_toolkit.clients.openai import AzureOpenAIConfig

    monkeypatch.setattr(toolkit.settings, "MI2_PREMIUM_API_KEY", key)
    monkeypatch.setattr(toolkit.settings, "CXRREPORTGEN_PREMIUM_API_KEY", key)
    monkeypatch.setattr(
        test_endpoints,
        "defaultConfig",
        AzureOpenAIConfig(endpoint="https://example.invalid", api_key=key),
    )
    monkeypatch.setattr(AzureOpenAIConfig, "_fetch_key", toolkit.forbidden)
    test_endpoints.print_configuration()
    output = capsys.readouterr().out
    expected = "(set)" if key else "(not set)"
    for name in (
        "MI2_PREMIUM_API_KEY",
        "CXRREPORTGEN_PREMIUM_API_KEY",
        "AZURE_OPENAI_API_KEY",
    ):
        assert f"{name}:\n    {expected}" in output
    assert "prefixAB" not in output and "suffixYZ" not in output
    if key:
        assert key not in output


def test_client_settings_and_explicit_value_only_manager(toolkit, monkeypatch):
    endpoint = "https://endpoint.region.inference.ml.azure.com/score"
    toolkit.settings._constants.update(
        MI2_MODEL_ENDPOINT=endpoint, MI2_MODEL_API_KEY="setting-key"
    )
    manager = toolkit.clients.MedImageInsightClient().endpoint_manager
    assert manager.target == endpoint
    assert manager.headers == {"Authorization": "Bearer setting-key"}

    monkeypatch.setattr(toolkit.settings, "get", toolkit.forbidden)
    client = toolkit.clients.MedImageInsightClient(
        "https://account.services.ai.azure.com/embed",
        api_key="override-key",
    )
    assert client.endpoint_manager.target == (
        "https://account.services.ai.azure.com/embed"
    )
    assert client.endpoint_manager.headers == {"Authorization": "Bearer override-key"}
    manager = toolkit.managers.ConnectionManager.from_endpoint(
        "https://account.services.ai.azure.com",
        credential=object(),
        api_key="override-key",
        api_path="/embed",
    )
    assert manager.target == "https://account.services.ai.azure.com/embed"
    assert manager.headers == {"Authorization": "Bearer override-key"}
    with pytest.raises(ValueError, match="Set an endpoint"):
        toolkit.clients.MedImageInsightClient("", api_key="override-key")


def test_api_key_required_and_invalid_key_not_disclosed(toolkit):
    endpoint = "https://account.services.ai.azure.com/score"
    api_key = "synthetic-secret\u200b"
    with pytest.raises(ValueError, match="API key"):
        toolkit.clients.MedImageInsightClient(endpoint)
    with pytest.raises(ValueError, match="ASCII API key") as error:
        toolkit.clients.MedImageInsightClient(endpoint, api_key=api_key)
    assert "synthetic-secret" not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://unknown.example/score",
        "https://synthetic-secret@account.services.ai.azure.com/score",
    ],
)
def test_invalid_target_is_secret_safe(toolkit, endpoint):
    api_key = "synthetic-secret"
    with pytest.raises(ValueError) as error:
        toolkit.clients.MedImageInsightClient(endpoint, api_key=api_key)
    assert "synthetic-secret" not in "".join(traceback.format_exception(error.value))
