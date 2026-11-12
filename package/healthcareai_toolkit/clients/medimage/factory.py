# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from healthcareai_toolkit import settings

from .cxrreportgen import CxrReportGenClient
from .cxrreportgen_premium import CxrReportGenPremiumClient
from .gigapath import GigaPathClient
from .gigatime import GigaTimeClient
from .medimageinsight import MedImageInsightClient
from .medimageinsight_premium import MedImageInsightPremiumClient
from .medimageparse import MedImageParseClient


_CLIENTS = {
    "mi2": {
        "opensource": MedImageInsightClient,
        "premium": MedImageInsightPremiumClient,
    },
    "cxrreportgen": {
        "opensource": CxrReportGenClient,
        "premium": CxrReportGenPremiumClient,
    },
    "mip": {"opensource": MedImageParseClient, "premium": None},
    "gigapath": {"opensource": GigaPathClient, "premium": None},
    "gigatime": {"opensource": GigaTimeClient, "premium": None},
}


def get_medimage_client(model, variant=None, **kwargs):
    """Select a medical-image client and forward kwargs unchanged to its constructor.

    An explicit 'opensource' or 'premium' variant selects an available class directly.
    Otherwise configured endpoint settings select available premium first, then opensource.
    Constructor kwargs do not affect automatic selection, and constructor errors
    propagate without falling back to another variant.
    """
    if not isinstance(model, str) or model not in _CLIENTS:
        raise ValueError("model not supported")
    variants = _CLIENTS[model]
    if variant is None:
        premium = variants.get("premium")
        opensource = variants.get("opensource")
        if premium is not None and settings.get(premium.ENDPOINT_NAME_SETTING):
            variant = "premium"
        elif opensource is not None and settings.get(opensource.ENDPOINT_NAME_SETTING):
            variant = "opensource"
        else:
            required_settings = " or ".join(
                client_class.ENDPOINT_NAME_SETTING
                for client_class in (premium, opensource)
                if client_class is not None
            )
            raise ValueError(f"You must set {required_settings}.")

    if variant not in ("premium", "opensource"):
        raise ValueError("variant must be 'opensource', 'premium', or None.")
    client_class = variants.get(variant)
    if client_class is None:
        raise ValueError(f"model {model!r} with variant {variant!r} not supported")
    return client_class(**kwargs)
