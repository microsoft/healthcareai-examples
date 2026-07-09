# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Self-contained thin MedImageInsight client.

Resolves the scoring URI and primary key from an AzureML online-endpoint
resource ID, then submits images and/or text to the MI2 scoring endpoint.

Config: set MI2_MODEL_ENDPOINT in your .env (or environment) to the full
AzureML online-endpoint resource ID:
  /subscriptions/<sub>/resourceGroups/<rg>/providers/
    Microsoft.MachineLearningServices/workspaces/<ws>/onlineEndpoints/<name>
"""

import base64
import logging
import re
from io import BytesIO
from itertools import zip_longest

import numpy as np
import requests
from PIL import Image

from errors import ToolRuntimeError

logger = logging.getLogger(__name__)


def _resolve_endpoint(
    resource_id: str, exclude_managed_identity: bool = False
) -> tuple[str, str]:
    # reason: optional dep; imported only when Azure endpoint setup is used.
    from azure.ai.ml import MLClient

    # reason: optional dep; imported only when Azure endpoint setup is used.
    from azure.identity import DefaultAzureCredential

    if not resource_id:
        raise ToolRuntimeError(
            "A non-empty AzureML online-endpoint resource ID is required."
        )

    match = re.match(
        r"/subscriptions/(.*)/resourceGroups/(.*)/providers/Microsoft.MachineLearningServices/workspaces/(.*)/onlineEndpoints/(.*)",
        resource_id,
        re.IGNORECASE,
    )
    if not match:
        raise ToolRuntimeError(
            "MI2_MODEL_ENDPOINT does not look like an AzureML online-endpoint "
            f"resource ID: {resource_id!r}. Expected the full resource ID: "
            "/subscriptions/<sub>/resourceGroups/<rg>/providers/"
            "Microsoft.MachineLearningServices/workspaces/<ws>/onlineEndpoints/<name>."
        )

    (
        subscription_id,
        resource_group_name,
        workspace_name,
        endpoint_name,
    ) = match.groups()

    credential = DefaultAzureCredential(
        exclude_managed_identity_credential=exclude_managed_identity
    )
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
    )

    # reason: optional dep; imported only when Azure endpoint setup is used.
    from azure.core.exceptions import AzureError, ClientAuthenticationError

    try:
        endpoint = ml_client.online_endpoints.get(name=endpoint_name)
        if endpoint.auth_mode != "key":
            raise ToolRuntimeError(
                f"auth_mode {endpoint.auth_mode!r} not supported; endpoint must use 'key' auth."
            )
        keys = ml_client.online_endpoints.get_keys(name=endpoint_name)
    except ClientAuthenticationError as exc:
        raise ToolRuntimeError(
            "Azure authentication failed while resolving the MI2 endpoint. Run `az login` "
            "(and `az account set --subscription <id>`), or check the credential / managed-identity setup."
        ) from exc
    except AzureError as exc:
        raise ToolRuntimeError(
            f"Could not resolve the MI2 endpoint from Azure: {exc} "
            "Verify the subscription, resource group, workspace, and endpoint name in "
            "MI2_MODEL_ENDPOINT, and that you have access."
        ) from exc
    return endpoint.scoring_uri, keys.primary_key


class MI2Client:
    """Thin client for the MedImageInsight scoring endpoint."""

    def __init__(self, scoring_uri: str, authorization: str) -> None:
        self.scoring_uri = scoring_uri
        self.authorization = authorization

    @classmethod
    def from_endpoint(
        cls, endpoint_id: str, exclude_managed_identity: bool = False
    ) -> "MI2Client":
        """Create a client from an AzureML online-endpoint resource ID."""
        if not endpoint_id:
            raise ToolRuntimeError(
                "endpoint_id must be a non-empty AzureML online-endpoint resource ID."
            )
        scoring_uri, key = _resolve_endpoint(
            endpoint_id, exclude_managed_identity=exclude_managed_identity
        )
        return cls(scoring_uri=scoring_uri, authorization=f"Bearer {key}")

    def _encode_image(self, file_path: str) -> str:
        """Load, normalize, and JPEG-encode a local path or http(s) URL for submission."""
        if file_path.startswith(("http://", "https://")):
            resp = requests.get(file_path, timeout=60)
            resp.raise_for_status()
            source = BytesIO(resp.content)
        else:
            source = file_path
        with Image.open(source) as src:
            img = src.convert("RGB")
        img = img.resize((512, 512), Image.BICUBIC)

        arr = np.asarray(img, dtype=np.float32)
        lo, hi = np.percentile(arr, (5.0, 95.0))
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def submit(self, image_list=None, text_list=None) -> list:
        """Submit images and/or text rows to the endpoint and return raw JSON."""
        image_list = [] if image_list is None else image_list
        text_list = [] if text_list is None else text_list

        encoded_images = [self._encode_image(path) for path in image_list]
        rows = [list(pair) for pair in zip_longest(encoded_images, text_list)]
        payload = {
            "input_data": {
                "columns": ["image", "text"],
                "index": list(range(len(rows))),
                "data": rows,
            },
            "params": {},
        }

        headers = {
            "Authorization": self.authorization,
            "Content-Type": "application/json",
        }
        resp = requests.post(
            self.scoring_uri,
            json=payload,
            headers=headers,
            timeout=120,
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code in (401, 403):
                raise ToolRuntimeError(
                    "MI2 request was not authorized. Re-run setup() or verify the endpoint key/auth configuration.",
                ) from exc
            if status_code == 404:
                raise ToolRuntimeError(
                    "MI2 endpoint was not found. Verify MI2_MODEL_ENDPOINT points to the right AzureML endpoint/deployment.",
                ) from exc
            if status_code == 429:
                raise ToolRuntimeError(
                    "MI2 endpoint is throttling requests. Retry in a moment.",
                ) from exc
            if status_code == 503:
                raise ToolRuntimeError(
                    "MI2 endpoint is starting or scaling. Retry in a moment.",
                ) from exc
            raise ToolRuntimeError(
                f"MI2 request failed (HTTP {status_code}). Retry, or check the endpoint and deployment health.",
            ) from exc
        return resp.json()
