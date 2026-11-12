# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import importlib
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock

import pytest


ROOT = "https://account.services.ai.azure.com"
AML = "https://endpoint.region.inference.ml.azure.com/score"
ARM = (
    "/subscriptions/sub/resourceGroups/rg/providers/"
    "Microsoft.MachineLearningServices/workspaces/ws/onlineEndpoints/endpoint"
)


@pytest.fixture
def factory_classes(toolkit, monkeypatch):
    from healthcareai_toolkit.clients.medimage import factory

    regular = Mock(ENDPOINT_NAME_SETTING="CUSTOM_REGULAR")
    premium = Mock(ENDPOINT_NAME_SETTING="UNRELATED_PREMIUM")
    monkeypatch.setattr(
        factory, "_CLIENTS", {"mi2": {"opensource": regular, "premium": premium}}
    )
    toolkit.settings._constants.update(CUSTOM_REGULAR=None, UNRELATED_PREMIUM=None)
    return regular, premium


def test_factory_class_owned_settings_and_unchanged_kwargs(toolkit, factory_classes):
    regular, premium = factory_classes
    toolkit.settings._constants.update(CUSTOM_REGULAR=AML, UNRELATED_PREMIUM=ROOT)
    kwargs = {
        "endpoint_name": ROOT + "/custom",
        "deployment_name": "my-deployment",
        "credential": object(),
        "api_key": "test-key",
        "arbitrary_option": object(),
    }
    assert toolkit.clients.get_medimage_client("mi2", **kwargs) is premium.return_value
    premium.assert_called_once_with(**kwargs)
    regular.assert_not_called()


@pytest.mark.parametrize(
    "setting,index", [("CUSTOM_REGULAR", 0), ("UNRELATED_PREMIUM", 1)]
)
def test_auto_uses_sole_configured_class(toolkit, factory_classes, setting, index):
    toolkit.settings._constants[setting] = ROOT
    assert (
        toolkit.clients.get_medimage_client("mi2")
        is factory_classes[index].return_value
    )


def test_auto_requires_settings_even_with_explicit_endpoint(toolkit, factory_classes):
    with pytest.raises(
        ValueError, match="You must set UNRELATED_PREMIUM or CUSTOM_REGULAR"
    ):
        toolkit.clients.get_medimage_client("mi2", endpoint_name=AML)
    factory_classes[0].assert_not_called()
    factory_classes[1].assert_not_called()


def test_explicit_variant_uses_kwargs_without_settings(toolkit, monkeypatch):
    monkeypatch.setattr(toolkit.settings, "get", toolkit.forbidden)
    deployment_name = " \tdeployment\x00\r\n\x7f "
    premium = toolkit.clients.get_medimage_client(
        "mi2",
        "premium",
        endpoint_name=AML,
        api_key="key",
        deployment_name=deployment_name,
    )
    assert premium.create_payload(text_list=["text"]) == {
        "model": deployment_name,
        "texts": ["text"],
    }
    opensource = toolkit.clients.get_medimage_client(
        "cxrreportgen", "opensource", endpoint_name=ROOT + "/score", api_key="key"
    )
    assert opensource.endpoint_manager.target == ROOT + "/score"


def test_invalid_preferred_class_does_not_fallback_but_explicit_opensource_works(
    toolkit,
):
    toolkit.settings._constants.update(
        MI2_MODEL_ENDPOINT=AML,
        MI2_MODEL_API_KEY="regular-key",
        MI2_PREMIUM_ENDPOINT=ROOT,
        MI2_PREMIUM_API_KEY="",
        MI2_PREMIUM_DEPLOYMENT_NAME="deployment",
    )
    with pytest.raises(ValueError, match="API key"):
        toolkit.clients.get_medimage_client("mi2")
    opensource = toolkit.clients.get_medimage_client("mi2", "opensource")
    assert opensource.endpoint_manager.headers == {
        "Authorization": "Bearer regular-key"
    }


def test_factory_rejects_unknown_model_and_variant(toolkit):
    with pytest.raises(ValueError, match="^model not supported$"):
        toolkit.clients.get_medimage_client("unknown")
    with pytest.raises(ValueError, match="variant"):
        toolkit.clients.get_medimage_client("mi2", "auto")
    with pytest.raises(ValueError, match="variant"):
        toolkit.clients.get_medimage_client("mi2", "regular")


@pytest.mark.parametrize(
    "model,client_name",
    [
        ("mip", "MedImageParseClient"),
        ("gigapath", "GigaPathClient"),
        ("gigatime", "GigaTimeClient"),
    ],
)
def test_factory_auto_selects_opensource_only_models(
    toolkit, monkeypatch, model, client_name
):
    client_class = getattr(toolkit.clients, client_name)
    constructor = Mock(return_value=None)
    monkeypatch.setattr(client_class, "__init__", constructor)
    setting = client_class.ENDPOINT_NAME_SETTING
    get_setting = Mock(side_effect={setting: AML}.__getitem__)
    monkeypatch.setattr(toolkit.settings, "get", get_setting)
    kwargs = {"endpoint_name": ROOT, "api_key": "key", "arbitrary_option": object()}
    client = toolkit.clients.get_medimage_client(model, **kwargs)
    assert isinstance(client, client_class)
    constructor.assert_called_once_with(**kwargs)
    get_setting.assert_called_once_with(setting)


def test_factory_opensource_only_unsupported_premium_and_missing_config(
    toolkit, monkeypatch
):
    monkeypatch.setattr(toolkit.settings, "get", toolkit.forbidden)
    with pytest.raises(
        ValueError, match="model 'mip' with variant 'premium' not supported"
    ):
        toolkit.clients.get_medimage_client("mip", "premium")
    setting = toolkit.clients.MedImageParseClient.ENDPOINT_NAME_SETTING
    get_setting = Mock(side_effect={setting: None}.__getitem__)
    monkeypatch.setattr(toolkit.settings, "get", get_setting)
    with pytest.raises(ValueError, match=f"^You must set {setting}\\.$"):
        toolkit.clients.get_medimage_client("mip")
    get_setting.assert_called_once_with(setting)


@pytest.mark.parametrize(
    "root,full_path",
    [(ROOT, "/providers/microsoft/v1/inference"), (ROOT + "/", "/swagger.json")],
)
def test_premium_env_setup_root_completion_and_full_route(
    toolkit, monkeypatch, root, full_path
):
    monkeypatch.setenv("CXRREPORTGEN_PREMIUM_ENDPOINT", root)
    monkeypatch.setenv("CXRREPORTGEN_PREMIUM_API_KEY", "setting-key")
    monkeypatch.setenv("CXRREPORTGEN_PREMIUM_DEPLOYMENT_NAME", "deployment")
    spec = importlib.util.spec_from_file_location(
        "fresh_settings", toolkit.settings.__file__
    )
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    monkeypatch.setattr(toolkit.settings, "_constants", fresh._constants)
    client = toolkit.clients.CxrReportGenPremiumClient()
    target = ROOT + "/providers/microsoft/v1/inference"
    assert client.deployment_name == "deployment"
    assert client.endpoint_manager.target == target
    assert client.endpoint_manager.headers == {"Authorization": "Bearer setting-key"}
    full_target = ROOT + full_path + "?api-version=2026-01-01"
    deployment_name = " \toverride\x00\r\n\x7f "
    full = toolkit.clients.CxrReportGenPremiumClient(
        full_target,
        object(),
        {"max_tokens": 32},
        {"engine": "pydicom"},
        {"min_max": (0, 100)},
        {"resize_size": (8, 8)},
        {"retries": 2},
        deployment_name=deployment_name,
        api_key="override-key",
    )
    assert full.endpoint_manager.target == full_target
    assert full.deployment_name == deployment_name
    assert full.endpoint_manager.headers == {"Authorization": "Bearer override-key"}
    assert full.params == {"max_tokens": 32}
    assert full.image_reader.engine == "pydicom"
    assert full.image_preprocessor.min_max == (0, 100)
    assert full.image_preprocessor.resize_size == (8, 8)
    with pytest.raises(ValueError, match="DEPLOYMENT_NAME"):
        toolkit.clients.CxrReportGenPremiumClient(deployment_name="")
    with pytest.raises(ValueError, match="full inference URL"):
        toolkit.clients.CxrReportGenClient(ROOT, api_key="key")


@pytest.mark.parametrize("root", [ROOT, ROOT + "/"])
def test_foundry_root_completion_preserves_query_and_fragment(toolkit, root):
    manager = toolkit.managers.ConnectionManager.from_endpoint(
        root + "?api-version=2026-01-01#section",
        api_key="key",
        api_path="/providers/microsoft/v2/embed",
    )
    assert manager.target == (
        ROOT + "/providers/microsoft/v2/embed?api-version=2026-01-01#section"
    )


@pytest.mark.parametrize(
    "client_name,default_deployment",
    [
        ("MedImageInsightPremiumClient", "MedImageInsight-Premium"),
        ("CxrReportGenPremiumClient", "CxrReportGen-Premium"),
    ],
)
def test_premium_deployment_defaults_and_overrides(
    toolkit, client_name, default_deployment
):
    client_class = getattr(toolkit.clients, client_name)
    kwargs = {"endpoint_name": ROOT, "api_key": "test-key"}
    assert client_class(**kwargs).deployment_name == default_deployment
    toolkit.settings._constants[client_class.DEPLOYMENT_NAME_SETTING] = ""
    assert client_class(**kwargs).deployment_name == default_deployment
    toolkit.settings._constants[client_class.DEPLOYMENT_NAME_SETTING] = "configured"
    assert client_class(**kwargs).deployment_name == "configured"
    assert (
        client_class(**kwargs, deployment_name="explicit").deployment_name == "explicit"
    )
    with pytest.raises(ValueError, match="DEPLOYMENT_NAME"):
        client_class(**kwargs, deployment_name="")


@pytest.fixture
def ml_client(toolkit, monkeypatch):
    client = Mock()
    client.online_endpoints = Mock(spec_set=["get", "get_keys"])
    client.online_endpoints.get.return_value = SimpleNamespace(
        scoring_uri=AML, auth_mode="key"
    )
    client.online_endpoints.get_keys.return_value = SimpleNamespace(
        primary_key="aml-key"
    )
    constructor = Mock(return_value=client)
    constructor.from_config.return_value = client
    monkeypatch.setattr(toolkit.managers, "MLClient", constructor)
    return constructor, client


def test_aml_id_looks_up_target_and_key_with_user_credential(
    toolkit, monkeypatch, ml_client
):
    constructor, management = ml_client
    monkeypatch.setenv("AZUREML_RUN_ID", "synthetic-run")
    monkeypatch.setenv("AZUREML_SERVICE_ENDPOINT", "https://component.invalid")
    toolkit.settings._constants["MI2_MODEL_API_KEY"] = ""
    credential = object()
    manager = toolkit.clients.MedImageInsightClient(ARM, credential).endpoint_manager
    constructor.assert_called_once_with(
        credential=credential,
        subscription_id="sub",
        resource_group_name="rg",
        workspace_name="ws",
    )
    management.online_endpoints.get.assert_called_once_with(name="endpoint")
    assert manager.target == AML
    assert manager.headers == {"Authorization": "Bearer aml-key"}
    with pytest.raises(ValueError, match="API key"):
        toolkit.clients.MedImageInsightClient(ARM, credential, api_key="")
    management.online_endpoints.get_keys.assert_called_once_with(name="endpoint")
    management.online_endpoints.get.return_value.auth_mode = "aad_token"
    with pytest.raises(ValueError, match="API-key scoring"):
        toolkit.clients.MedImageInsightClient(ARM, credential)


def test_aml_local_name_completes_lookup_with_default_credential(
    toolkit, monkeypatch, ml_client
):
    constructor, management = ml_client
    monkeypatch.setenv("AZUREML_RUN_ID", "synthetic-run")
    credential = object()
    default = Mock(return_value=credential)
    monkeypatch.setattr(toolkit.identity, "DefaultAzureCredential", default)
    monkeypatch.setattr(toolkit.settings, "DISABLE_MANAGED_IDENTITY", True)
    client = toolkit.clients.MedImageInsightClient("existing-name")
    default.assert_called_once_with(exclude_managed_identity_credential=True)
    constructor.from_config.assert_called_once_with(credential)
    management.online_endpoints.get.assert_called_once_with(name="existing-name")
    management.online_endpoints.get_keys.assert_called_once_with(name="existing-name")
    assert client.endpoint_manager.target == AML
    assert client.endpoint_manager.headers == {"Authorization": "Bearer aml-key"}


def test_mip_forwards_options_to_shared_base(toolkit, monkeypatch):
    retry_client = Mock(wraps=toolkit.base.RetryClient)
    monkeypatch.setattr(toolkit.base, "RetryClient", retry_client)
    client = toolkit.clients.MedImageParseClient(
        AML,
        object(),
        "pydicom",
        "ct-lung",
        {"option": 1},
        {"min_max": (0, 100)},
        {"retries": 2},
        api_key="direct-key",
    )
    assert client.create_payload()["params"] == {"option": 1}
    assert client.image_reader.engine == "pydicom"
    assert client.image_preprocessor.min_max == [-1000, 1000]
    assert client.image_preprocessor.resize_size == 1024
    assert client.endpoint_manager.headers == {"Authorization": "Bearer direct-key"}
    retry_client.assert_called_once_with(
        retry_params={"retries": 2}, logger=client.logger
    )


def test_aml_url_preserves_positional_options_without_management_setup(toolkit):
    client = toolkit.clients.MedImageInsightClient(
        AML,
        object(),
        {"option": 1},
        {"engine": "pydicom"},
        {"min_max": (0, 100)},
        {"resize_size": (8, 8)},
        {"retries": 2},
        api_key="direct-key",
    )
    assert client.params == {"option": 1}
    assert client.image_reader.engine == "pydicom"
    assert client.image_preprocessor.resize_size == (8, 8)
    assert client.image_preprocessor.min_max == (0, 100)
    assert "direct-key" not in repr(client.endpoint_manager)
    client.retry_client.submit_payload = Mock(return_value=[])
    assert client.submit(text_list=["text"]) == []
    client.retry_client.submit_payload.assert_called_once_with(
        client.create_payload(text_list=["text"]),
        target=AML,
        headers={"Authorization": "Bearer direct-key"},
    )
