# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import base64
from io import BytesIO
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image


def test_mi2_cli_accepts_nested_embeddings(mi2, toolkit, monkeypatch):
    from healthcareai_toolkit.cli import test_endpoints

    monkeypatch.setattr(
        toolkit.settings, "MI2_PREMIUM_ENDPOINT", "https://example.invalid"
    )
    monkeypatch.setattr(test_endpoints, "MedImageInsightPremiumClient", lambda: mi2)
    monkeypatch.setattr(test_endpoints.glob, "glob", lambda pattern: ["synthetic.dcm"])
    monkeypatch.setattr(test_endpoints.os.path, "isfile", lambda path: True)
    monkeypatch.setattr(mi2, "_read_and_encode_choice", lambda image: "encoded")
    mi2.retry_client.submit_payload = Mock(
        side_effect=[
            {"embeddings": {"float": [[0.1, 0.2]]}},
            {"embeddings": {"float": [[0.3, 0.4]]}, "texts": ["synthetic"]},
        ]
    )
    assert test_endpoints.test_medimageinsight_premium_endpoint(quiet=True) is True
    assert mi2.retry_client.submit_payload.call_count == 2


@pytest.fixture
def image():
    return np.arange(16 * 24, dtype=np.uint16).reshape(16, 24)


@pytest.fixture
def mi2(toolkit):
    return toolkit.clients.MedImageInsightPremiumClient(
        "https://account.services.ai.azure.com",
        api_key="key",
        deployment_name="mi2-deployment",
    )


@pytest.fixture
def cxr(toolkit):
    return toolkit.clients.CxrReportGenPremiumClient(
        "https://account.services.ai.azure.com",
        api_key="key",
        deployment_name="cxr-deployment",
    )


def test_mi2_image_payload_preprocessing_and_decode(mi2, image, toolkit, monkeypatch):
    monkeypatch.setitem(
        toolkit.clients.MedImageInsightClient.default_normalization,
        "percentiles",
        (0.1, 0.9),
    )
    monkeypatch.setitem(
        toolkit.clients.MedImageInsightClient.default_transform, "resize_size", (8, 8)
    )
    assert mi2.default_normalization == {"percentiles": (0.01, 0.99)}
    assert mi2.default_transform == {
        "pad_to_square": False,
        "resize_size": (480, 480),
        "resize_edge": False,
    }
    source = BytesIO()
    Image.fromarray(image.astype(np.uint8)).save(source, format="JPEG")
    payload = mi2.create_payload(image_list=[source.getvalue()])
    prefix, data = payload["images"][0].split(",", 1)
    assert prefix == "data:image/png;base64"
    assert payload == {
        "model": "mi2-deployment",
        "images": [f"data:image/png;base64,{data}"],
    }
    assert not any(char.isspace() for char in data)
    with Image.open(BytesIO(base64.b64decode(data))) as transformed:
        assert transformed.format == "PNG"
        assert transformed.size == (480, 480)
        assert transformed.mode == "RGB"
    assert mi2.decode_response(
        {"embeddings": {"float": [[0.1, 0.2]]}, "id": "request"}
    ) == [
        {
            "embeddings": {"float": [0.1, 0.2]},
            "image_features": [[0.1, 0.2]],
            "id": "request",
        }
    ]
    assert mi2.endpoint_manager.target.endswith("/providers/microsoft/v2/embed")


def test_mi2_text_and_file_payloads_and_response_metadata(mi2, tmp_path):
    assert mi2.create_payload(text_list=["clinical text"]) == {
        "model": "mi2-deployment",
        "texts": ["clinical text"],
    }
    text_file = tmp_path / "prompt.txt"
    text_file.write_text("clinical text", encoding="utf-8")
    mi2.params = {"embedding_types": ["float", "binary"]}
    assert mi2.create_payload(text_file_list=[text_file]) == {
        "model": "mi2-deployment",
        "texts": ["clinical text"],
        **mi2.params,
    }
    response = {
        "texts": ["clinical text"],
        "embeddings": {"float": [[0.1, 0.2]], "binary": [[1]]},
        "meta": {"api_version": "test"},
        "usage": {"billed_units": 1},
    }
    assert mi2.decode_response(response) == [
        {
            "text": ["clinical text"],
            "text_features": [[0.1, 0.2]],
            "embeddings": {"float": [0.1, 0.2], "binary": [1]},
            "meta": response["meta"],
            "usage": response["usage"],
        }
    ]
    assert mi2.decode_response({"embeddings": {"binary": [[1]]}}) == [
        {"embeddings": {"binary": [1]}}
    ]


def test_mi2_scaling_factor_only_when_returned(mi2):
    response = {
        "texts": ["first", "second"],
        "embeddings": {"float": [[0.1, 0.2], [0.3, 0.4]]},
    }
    rows = mi2.decode_response(response)
    assert all("scaling_factor" not in row for row in rows)
    assert "scaling_factor" not in response
    for scaling_factor in (2.0, 0.0, None):
        supplied = {**response, "scaling_factor": scaling_factor}
        assert [row["scaling_factor"] for row in mi2.decode_response(supplied)] == [
            scaling_factor,
            scaling_factor,
        ]
        assert supplied == {**response, "scaling_factor": scaling_factor}
    assert mi2.create_payload(text_list=["first"]) == {
        "model": "mi2-deployment",
        "texts": ["first"],
    }


@pytest.mark.parametrize("field", ["image_list", "text_list", "text_file_list"])
def test_mi2_rejects_multiple_inputs_before_processing_or_sending(
    mi2, monkeypatch, field
):
    encode = Mock(side_effect=AssertionError("must not read image"))
    read = Mock(side_effect=AssertionError("must not read file"))
    monkeypatch.setattr(mi2, "_read_and_encode_choice", encode)
    monkeypatch.setattr(mi2, "_read_text_file", read)
    monkeypatch.setattr(
        mi2,
        "create_payload",
        Mock(side_effect=AssertionError("must not build payload")),
    )
    mi2.retry_client.submit_payload = Mock()
    with pytest.raises(ValueError, match="one input"):
        mi2.submit(**{field: ["first", "second"]})
    encode.assert_not_called()
    read.assert_not_called()
    mi2.create_payload.assert_not_called()
    mi2.retry_client.submit_payload.assert_not_called()


def test_mi2_rejects_mixed_empty_and_unsupported_inputs(mi2, monkeypatch):
    monkeypatch.setattr(
        mi2,
        "_read_and_encode_choice",
        Mock(side_effect=AssertionError("must not read image")),
    )
    monkeypatch.setattr(
        mi2, "_read_text_file", Mock(side_effect=AssertionError("must not read file"))
    )
    mi2.retry_client.submit_payload = Mock(
        return_value={"embeddings": {"float": [[1]]}}
    )
    with pytest.raises(ValueError, match="exactly one nonempty"):
        mi2.submit(image_list=["not-read"], text_list=["text"])
    with pytest.raises(ValueError, match="exactly one nonempty"):
        mi2.submit(text_list=["text"], text_file_list=["not-read"])
    with pytest.raises(ValueError, match="nonempty"):
        mi2.submit()
    with pytest.raises(ValueError, match="nonempty"):
        mi2.submit(text_list=[])
    with pytest.raises(ValueError, match="list or tuple"):
        mi2.submit(text_list="text")
    mi2.params = {"model": "override", "embedding_types": ["unsupported"]}
    mi2.submit(image_list=[], text_list=("text",))
    assert mi2.retry_client.submit_payload.call_args.args[0] == {
        "model": "override",
        "embedding_types": ["unsupported"],
        "texts": ["text"],
    }


def test_cxr_minimal_prior_payload_and_decode(cxr, image, toolkit, monkeypatch):
    monkeypatch.setitem(
        toolkit.clients.CxrReportGenClient.default_normalization,
        "percentiles",
        (0.1, 0.9),
    )
    assert cxr.default_normalization == {"percentiles": (0.01, 0.99)}
    minimal = cxr.create_payload(image)
    encoded = minimal["current_image"]
    assert minimal == {"model": "cxr-deployment", "current_image": encoded}
    assert not encoded.startswith("data:")
    with Image.open(BytesIO(base64.b64decode(encoded))) as result:
        assert result.format == "PNG"
        assert result.size == (24, 16)
        assert np.asarray(result).max() > np.asarray(result).min()
    cxr.params = {
        "max_tokens": 512,
        "temperature": 0.2,
        "text_normalize_strip": "prior",
    }
    payload = cxr.create_payload(
        frontal_image=image,
        indication="Cough",
        technique="AP frontal",
        comparison="Prior study",
        prior_image=image,
        prior_report="Prior findings",
    )
    assert payload == {
        **minimal,
        "indication": "Cough",
        "technique": "AP frontal",
        "comparison": "Prior study",
        "prior_image": encoded,
        "prior_report": "Prior findings",
        **cxr.params,
    }
    response = {
        "findings": "Draft findings",
        "model": "served-model",
        "usage": {"total_tokens": 12},
    }
    assert cxr.decode_response(response) == [{**response, "output": "Draft findings"}]
    cxr.retry_client.submit_payload = Mock(return_value=response)
    assert cxr.submit(
        frontal_image=image, prior_image=image, prior_report=("Prior findings",)
    ) == [{**response, "output": "Draft findings"}]
    submitted = cxr.retry_client.submit_payload.call_args.args[0]
    assert submitted["current_image"] == encoded
    assert submitted["prior_image"] == encoded
    assert submitted["prior_report"] == "Prior findings"


def test_cxr_rejects_multiple_studies_and_unsupported_context(cxr, image, monkeypatch):
    monkeypatch.setattr(
        cxr,
        "_read_and_encode_choice",
        Mock(side_effect=AssertionError("must not process")),
    )
    monkeypatch.setattr(
        cxr,
        "create_payload",
        Mock(side_effect=AssertionError("must not build payload")),
    )
    cxr.retry_client.submit_payload = Mock()
    with pytest.raises(ValueError, match="one study"):
        cxr.submit(frontal_image=[image, image])
    with pytest.raises(ValueError, match="indication.*one study"):
        cxr.submit(frontal_image=image, indication=["first", "second"])
    with pytest.raises(ValueError, match="prior_image.*one study"):
        cxr.submit(
            frontal_image=image,
            prior_image=[image, image],
            prior_report=["first", "second"],
        )
    with pytest.raises(ValueError, match="frontal_image is required"):
        cxr.submit()
    with pytest.raises(ValueError, match="together"):
        cxr.submit(frontal_image=image, prior_image=image)
    with pytest.raises(ValueError, match="lateral_image"):
        cxr.submit(frontal_image=image, lateral_image=image)
    with pytest.raises(ValueError, match="indication"):
        cxr.submit(frontal_image=image, indication="")
    cxr.params = {"model": "override"}
    with pytest.raises(ValueError, match="params"):
        cxr.submit(frontal_image=image)
    cxr.create_payload.assert_not_called()
    cxr._read_and_encode_choice.assert_not_called()
    cxr.retry_client.submit_payload.assert_not_called()


def test_premium_submitters_send_singletons_and_reject_larger_batches(mi2, cxr, image):
    mi2.retry_client.submit_payload = Mock(
        return_value={"embeddings": {"float": [[1]]}}
    )
    cxr.retry_client.submit_payload = Mock()
    with pytest.raises(ValueError, match="batch_size=1"):
        mi2.create_submitter(n_jobs=1, batch_size=2).submit(
            text_list=["first", "second"]
        )
    with pytest.raises(ValueError, match="batch_size=1"):
        cxr.create_submitter(n_jobs=1, batch_size=2).submit(
            frontal_image=[image, image]
        )
    mi2.retry_client.submit_payload.assert_not_called()
    cxr.retry_client.submit_payload.assert_not_called()
    indexes, rows = mi2.create_submitter(n_jobs=1).submit(text_list=["first", "second"])
    assert indexes == [0, 1]
    assert len(rows) == 2
    assert [
        call.args[0] for call in mi2.retry_client.submit_payload.call_args_list
    ] == [
        {"model": "mi2-deployment", "texts": ["first"]},
        {"model": "mi2-deployment", "texts": ["second"]},
    ]

    def response(payload, **kwargs):
        assert isinstance(payload["current_image"], str)
        assert "input_data" not in payload
        return {"findings": payload["indication"], "model": "served-model", "usage": {}}

    cxr.retry_client.submit_payload = Mock(side_effect=response)
    indexes, rows = cxr.create_submitter(n_jobs=1).submit(
        frontal_image=[image, image],
        indication=["first", "second"],
        prior_image=[None, image],
        prior_report=[None, "prior findings"],
    )
    assert indexes == [0, 1]
    assert [row["output"] for row in rows] == ["first", "second"]
    assert cxr.retry_client.submit_payload.call_count == 2
    payloads = [call.args[0] for call in cxr.retry_client.submit_payload.call_args_list]
    assert "prior_image" not in payloads[0]
    assert "prior_report" not in payloads[0]
    assert "prior_image" in payloads[1]
    assert payloads[1]["prior_report"] == "prior findings"


def test_regular_clients_keep_original_scoring_schema(toolkit, image):
    endpoint = "https://account.services.ai.azure.com/score"
    mi2 = toolkit.clients.MedImageInsightClient(
        endpoint, api_key="test-key", params={"option": 1}
    )
    payload = mi2.create_payload(image_list=[image], text_list=["text"])
    assert payload["input_data"]["columns"] == ["image", "text"]
    assert payload["input_data"]["index"] == [0]
    assert not payload["input_data"]["data"][0][0].startswith("data:")
    assert payload["params"] == {"option": 1}
    response = [{"image_features": [1], "text_features": [2], "scaling_factor": 3}]
    assert mi2.decode_response(response) is response
    mi2.retry_client.submit_payload = Mock(return_value=response)
    assert mi2.submit(image_list=[image], text_list=["text"]) == response
    assert mi2.retry_client.submit_payload.call_args.args[0] == payload
    cxr = toolkit.clients.CxrReportGenClient(endpoint, api_key="test-key")
    payload = cxr.create_payload(image, "", "")
    assert payload["input_data"]["columns"] == [
        "frontal_image",
        "indication",
        "technique",
        "comparison",
    ]
    assert payload["input_data"]["data"][0][1:] == ["", "", "None"]
    assert payload["params"] == {}
    assert cxr.decode_response([{"output": '["findings"]'}]) == [
        {"output": ["findings"]}
    ]
    cxr.retry_client.submit_payload = Mock(return_value=[{"output": '["findings"]'}])
    assert cxr.submit(frontal_image=image, indication="", technique="") == [
        {"output": ["findings"]}
    ]
    assert cxr.retry_client.submit_payload.call_args.args[0] == payload
