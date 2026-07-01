#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Sanity-check the MedImageInsight MCP server tools.

Runs each MCP tool once against a single test image so you can confirm the
server is wired up correctly before pointing an agent at it. If a tool fails
here, the problem is the server/endpoint config -- not your agent harness.

Without --image, only the label lookup and a text-embedding liveness check run
(enough to confirm the endpoint answers). Pass --image (a local path or http(s)
URL) to also exercise the image classification tools.

Run from this directory:

    python smoketest.py --quiet                   # default env mode
    python smoketest.py --endpoint <resource-id>
    python smoketest.py --env-file .env
    python smoketest.py --image path/to/image.png
    python smoketest.py --image https://host/cxr.png
"""

import argparse
import os
import traceback

import adapter
import config
from mi2_client import MI2Client
import numpy as np
import zero_shot


class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def cli_print(message: str) -> None:
    """Print user-facing test harness output."""
    print(message)  # slopcop: ignore[no-print]


def configure_client(args: argparse.Namespace) -> MI2Client:
    """Build an MI2 client using the shared config resolution."""
    kwargs = {"MI2_MODEL_ENDPOINT": args.endpoint} if args.endpoint else {}
    return config.build_client(
        config.resolve_settings(env_file=args.env_file, **kwargs)
    )


def test_zeroshot_label_examples(quiet: bool = False) -> bool:
    """Exercise the zeroshot_label_examples tool (local, no endpoint)."""
    cli_print(f"\n{Colors.BLUE}Testing zeroshot_label_examples...{Colors.END}")
    try:
        modalities = zero_shot.label_examples(group="modality")
        matches = zero_shot.label_examples(regex="pneumonia")
        if not modalities or not matches:
            cli_print(f"{Colors.RED}✗ Expected non-empty label results{Colors.END}")
            return False
        cli_print(
            f"{Colors.GREEN}✓ Returned {len(modalities)} modalities, "
            f"{len(matches)} pneumonia labels{Colors.END}"
        )
        if not quiet:
            cli_print(f"  modalities: {modalities[:5]}")
            cli_print(f"  pneumonia[0]: {matches[0]}")
        return True
    except Exception as e:  # slopcop: ignore[no-broad-except]
        cli_print(f"{Colors.RED}✗ Tool failed: {e}{Colors.END}")
        traceback.print_exc()
        return False


def test_endpoint_liveness(mi2_client: MI2Client, quiet: bool = False) -> bool:
    """Confirm the endpoint answers by fetching a text embedding (no image needed)."""
    cli_print(
        f"\n{Colors.BLUE}Testing endpoint liveness (text embedding)...{Colors.END}"
    )
    try:
        result = mi2_client.submit(text_list=["x-ray chest"])
        feats = np.squeeze(np.array(result[0]["text_features"]))
        if feats.ndim != 1 or feats.size == 0:
            cli_print(f"{Colors.RED}✗ Endpoint returned no text embedding{Colors.END}")
            return False
        cli_print(
            f"{Colors.GREEN}✓ Endpoint returned a {feats.size}-dim text "
            f"embedding{Colors.END}"
        )
        return True
    except Exception as e:  # slopcop: ignore[no-broad-except]
        cli_print(f"{Colors.RED}✗ Endpoint liveness failed: {e}{Colors.END}")
        traceback.print_exc()
        return False


def test_zeroshot_classify(
    test_image: str,
    mi2_client: MI2Client,
    quiet: bool = False,
) -> bool:
    """Exercise the zeroshot_classify tool against one image."""
    cli_print(f"\n{Colors.BLUE}Testing zeroshot_classify...{Colors.END}")
    try:
        labels = [
            "x-ray chest anteroposterior Pneumonia",
            "x-ray chest anteroposterior No Finding",
        ]
        probs = zero_shot.classify(test_image, labels, mi2_client=mi2_client)
        if set(probs) != set(labels):
            cli_print(
                f"{Colors.RED}✗ Result keys do not match input labels{Colors.END}"
            )
            return False
        if abs(sum(probs.values()) - 1.0) > 1e-3:
            cli_print(f"{Colors.RED}✗ Probabilities do not sum to 1{Colors.END}")
            return False
        cli_print(
            f"{Colors.GREEN}✓ Classified into {len(probs)} labels "
            f"(probabilities sum to 1){Colors.END}"
        )
        if not quiet:
            for label, p in sorted(probs.items(), key=lambda kv: -kv[1]):
                cli_print(f"  {p:.4f}  {label}")
        return True
    except Exception as e:  # slopcop: ignore[no-broad-except]
        cli_print(f"{Colors.RED}✗ Tool failed: {e}{Colors.END}")
        traceback.print_exc()
        return False


def test_adapter_classify(
    test_image: str,
    mi2_client: MI2Client,
    quiet: bool = False,
) -> bool:
    """Exercise the adapter_classify tool against one image."""
    cli_print(f"\n{Colors.BLUE}Testing adapter_classify...{Colors.END}")
    try:
        result = adapter.classify(test_image, mi2_client=mi2_client)
    except Exception as e:  # slopcop: ignore[no-broad-except]
        cli_print(f"{Colors.RED}✗ Tool failed: {e}{Colors.END}")
        traceback.print_exc()
        return False
    valid = {"positive", "possible", "negative"}
    if not result or not set(result.values()) <= valid:
        cli_print(f"{Colors.RED}✗ Unexpected adapter output{Colors.END}")
        return False
    cli_print(
        f"{Colors.GREEN}✓ Adapter returned {len(result)} per-label "
        f"calls{Colors.END}"
    )
    if not quiet:
        for label, call in result.items():
            cli_print(f"  {call:9s}  {label}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sanity-check the MedImageInsight MCP server tools."
    )
    parser.add_argument(
        "--image",
        default=None,
        help="local path or http(s) URL of a test PNG/JPEG; omit to run text-only liveness",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="print less detail per tool"
    )
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument(
        "--endpoint",
        metavar="RESOURCE_ID",
        help="AzureML online-endpoint resource ID for MI2Client.from_endpoint",
    )
    config_group.add_argument(
        "--env-file",
        metavar="PATH",
        help="path to a .env file containing MI2_MODEL_ENDPOINT",
    )
    args = parser.parse_args()

    try:
        mi2_client = configure_client(args)
    except Exception as exc:  # slopcop: ignore[no-broad-except]
        if args.endpoint:
            cli_print(
                f"{Colors.RED}✗ MedImageInsight configuration failed from "
                f"--endpoint ({type(exc).__name__}){Colors.END}"
            )
        else:
            cli_print(
                f"{Colors.RED}✗ MedImageInsight configuration failed: "
                f"{exc}{Colors.END}"
            )
        traceback.print_exc()
        return 1

    test_image = args.image
    is_url = bool(test_image) and test_image.startswith(("http://", "https://"))
    if test_image and not is_url and not os.path.exists(test_image):
        cli_print(
            f"{Colors.RED}✗ Image not found: {test_image}. Pass a local path or "
            f"http(s) URL.{Colors.END}"
        )
        return 1

    results = {
        "zeroshot_label_examples": test_zeroshot_label_examples(args.quiet),
    }
    if test_image:
        cli_print(f"{Colors.GREEN}✓ Using test image: {test_image}{Colors.END}")
        results["zeroshot_classify"] = test_zeroshot_classify(
            test_image, mi2_client, args.quiet
        )
        results["adapter_classify"] = test_adapter_classify(
            test_image, mi2_client, args.quiet
        )
    else:
        cli_print(
            f"{Colors.YELLOW}No --image given; running text-only liveness "
            f"check.{Colors.END}"
        )
        results["endpoint_liveness"] = test_endpoint_liveness(mi2_client, args.quiet)

    cli_print(f"\n{Colors.BOLD}Summary{Colors.END}")
    failed = 0
    for name, ok in results.items():
        if ok:
            cli_print(f"  {Colors.GREEN}✓ {name}{Colors.END}")
        else:
            cli_print(f"  {Colors.RED}✗ {name}{Colors.END}")
            failed += 1

    if failed:
        cli_print(f"\n{Colors.RED}{failed} tool(s) failed.{Colors.END}")
        return 1
    cli_print(f"\n{Colors.GREEN}All tools wired up correctly.{Colors.END}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
