# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from healthcareai_toolkit import settings

from .medimagebase import MedImageBaseClient


class CxrReportGenPremiumClient(MedImageBaseClient):
    """Generate draft findings for one frontal study, optionally with a paired prior.

    Uses the regular frontal_image/clinical argument names. Singleton lists are
    accepted for the shared ParallelSubmitter. submit rejects multi-study lists
    and lateral images before payload processing.
    params holds the documented generation/text-normalization options, flattened
    into the request. Returns [{findings, model, usage, ..., output}], with output
    an alias of the findings string (not the regular model's parsed report array).
    Findings require qualified review. No experimental DICOM passthrough is added.
    """

    ENDPOINT_NAME_SETTING = "CXRREPORTGEN_PREMIUM_ENDPOINT"
    API_KEY_SETTING = "CXRREPORTGEN_PREMIUM_API_KEY"
    DEPLOYMENT_NAME_SETTING = "CXRREPORTGEN_PREMIUM_DEPLOYMENT_NAME"
    DEFAULT_DEPLOYMENT_NAME = "CxrReportGen-Premium"
    API_PATH = "/providers/microsoft/v1/inference"
    default_normalization = {"percentiles": (0.01, 0.99)}
    _PARAMS = {
        "max_tokens",
        "temperature",
        "text_normalize_unicode",
        "text_normalize_whitespace",
        "text_normalize_strip",
    }

    def __init__(self, *args, deployment_name=None, **kwargs):
        if deployment_name is None:
            deployment_name = (
                settings.get(self.DEPLOYMENT_NAME_SETTING)
                or self.DEFAULT_DEPLOYMENT_NAME
            )
        if deployment_name is None or deployment_name == "":
            raise ValueError(
                f"Set {self.DEPLOYMENT_NAME_SETTING} or pass deployment_name."
            )
        self.deployment_name = deployment_name
        super().__init__(*args, **kwargs)

    def _validate_kwargs(
        self,
        frontal_image=None,
        indication=None,
        technique=None,
        comparison=None,
        lateral_image=None,
        prior_image=None,
        prior_report=None,
    ):
        fields = {
            "frontal_image": frontal_image,
            "indication": indication,
            "technique": technique,
            "comparison": comparison,
            "lateral_image": lateral_image,
            "prior_image": prior_image,
            "prior_report": prior_report,
        }
        for name, value in fields.items():
            if isinstance(value, (list, tuple)):
                if len(value) != 1:
                    raise ValueError(
                        f"{name}: CXR Premium accepts one study per request; use batch_size=1."
                    )
                fields[name] = value[0]
        if fields["lateral_image"] is not None:
            raise ValueError(
                "CXR Premium does not accept lateral_image; supply only the frontal study."
            )
        if fields["frontal_image"] is None:
            raise ValueError("frontal_image is required.")
        if (fields["prior_image"] is None) != (fields["prior_report"] is None):
            raise ValueError("prior_image and prior_report must be supplied together.")
        if set(self.params) - self._PARAMS:
            raise ValueError(
                "CXR Premium params supports only generation and text-normalization options."
            )
        for name in ("indication", "technique", "comparison", "prior_report"):
            value = fields[name]
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{name} must be nonempty text or omitted.")

    def create_payload(
        self,
        frontal_image,
        indication=None,
        technique=None,
        comparison=None,
        lateral_image=None,
        prior_image=None,
        prior_report=None,
    ):
        frontal_image = (
            frontal_image[0]
            if isinstance(frontal_image, (list, tuple))
            else frontal_image
        )
        prior_image = (
            prior_image[0] if isinstance(prior_image, (list, tuple)) else prior_image
        )
        context = {
            "indication": (
                indication[0] if isinstance(indication, (list, tuple)) else indication
            ),
            "technique": (
                technique[0] if isinstance(technique, (list, tuple)) else technique
            ),
            "comparison": (
                comparison[0] if isinstance(comparison, (list, tuple)) else comparison
            ),
            "prior_report": (
                prior_report[0]
                if isinstance(prior_report, (list, tuple))
                else prior_report
            ),
        }
        payload = {
            "model": self.deployment_name,
            "current_image": self._read_and_encode_choice(frontal_image),
            **{name: value for name, value in context.items() if value is not None},
            **self.params,
        }
        if prior_image is not None:
            payload["prior_image"] = self._read_and_encode_choice(prior_image)
        return payload

    def decode_response(self, response):
        if not isinstance(response, dict) or not isinstance(
            response.get("findings"), str
        ):
            raise ValueError("Expected a findings string in the CXR Premium response.")
        return [{**response, "output": response["findings"]}]
