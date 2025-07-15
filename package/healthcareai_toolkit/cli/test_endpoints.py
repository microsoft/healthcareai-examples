#!/usr/bin/env python3
"""
Test healthcare AI model endpoints connectivity and functionality.

This command tests the deployed model endpoints to ensure they are accessible
and responding correctly.
"""

import sys
import os
import glob
from pathlib import Path
import argparse
import traceback
from typing import Optional
import numpy as np

# Import healthcare AI toolkit components
from healthcareai_toolkit import settings
from healthcareai_toolkit.clients.openai import create_openai_client

# Color codes for terminal output
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"  # End formatting


from healthcareai_toolkit.clients import (
    MedImageInsightClient,
    MedImageParseClient,
    CxrReportGenClient,
    GigaPathClient,
)


def pretty_print_array(array, display_name="Vector"):
    """Pretty print arrays with truncation for readability."""
    array = np.array(array)

    if array.ndim < 2:
        array = array.reshape(1, -1)

    def format_row(row):
        return np.array2string(
            row,
            separator=", ",
            precision=3,
            threshold=10,
            edgeitems=3,
            formatter={"float_kind": lambda x: f"{x:.3f}"},
        )

    def format_array(arr):
        if arr.ndim == 1:
            return [format_row(arr)]
        top = format_array(arr[0])
        bottom = format_array(arr[-1])
        top[0] = "[" + top[0]
        bottom[-1] = bottom[-1] + "]"
        return top + ["..."] + bottom

    for i, arr in enumerate(array):
        print(f"  {display_name} {i} (shape: {arr.shape}, dtype: {arr.dtype})")
        print("\n".join(f"    {l}" for l in format_array(arr)))
        if i > 3:
            break


def pretty_print_response(response):
    """Pretty print API responses."""
    if isinstance(response, dict):
        for key, value in response.items():
            pretty_print_array(value, display_name=key)
    elif isinstance(response, list):
        for i, item in enumerate(response):
            print(f"  == Response {i} ==")
            pretty_print_response(item)
    else:
        print(f"  {response}")


def test_medimageinsight_endpoint(quiet: bool = False) -> Optional[bool]:
    """Test MedImageInsight endpoint connectivity."""
    print(f"\n{Colors.BLUE}Testing MedImageInsight endpoint...{Colors.END}")

    # Check if endpoint is configured
    if not settings.MI2_MODEL_ENDPOINT:
        print(
            f"{Colors.YELLOW}⚠ MI2_MODEL_ENDPOINT not configured - skipping test{Colors.END}"
        )
        return None

    try:
        # Find test data
        data_root = settings.DATA_ROOT
        input_folder = os.path.join(
            data_root, "medimageinsight-classification", "images"
        )

        if not os.path.exists(input_folder):
            print(f"{Colors.YELLOW}⚠ Test data not found at {input_folder}{Colors.END}")
            print(
                f"{Colors.GREEN}✓ Skipping functional test (no test data){Colors.END}"
            )
            return True

        image_files = list(glob.glob(input_folder + "/*.dcm"))
        if not image_files:
            print(
                f"{Colors.YELLOW}⚠ No DICOM files found in {input_folder}{Colors.END}"
            )
            print(
                f"{Colors.GREEN}✓ Skipping functional test (no test data){Colors.END}"
            )
            return True

        test_image = image_files[0]
        print(
            f"{Colors.GREEN}✓ Found test image: {os.path.basename(test_image)}{Colors.END}"
        )

        # Test the endpoint
        client = MedImageInsightClient()
        response = client.submit(
            image_list=[test_image],
            text_list=["x-ray chest anteroposterior No Finding"],
        )

        if not all(
            key in response[0]
            for key in ["image_features", "text_features", "scaling_factor"]
        ):
            print(f"{Colors.RED}✗ Response does not contain expected keys{Colors.END}")
            return False

        print(f"{Colors.GREEN}✓ Endpoint responded with expected format{Colors.END}")

        if not quiet:
            pretty_print_response(response)

        return True

    except Exception as e:
        print(f"{Colors.RED}✗ Endpoint test failed: {str(e)}{Colors.END}")
        traceback.print_exc()
        return False


def test_medimageparse_endpoint(quiet: bool = False) -> bool:
    """Test MedImageParse endpoint connectivity."""
    print(f"\n{Colors.BLUE}Testing MedImageParse endpoint...{Colors.END}")

    # Check if endpoint is configured
    if not settings.MIP_MODEL_ENDPOINT:
        print(
            f"{Colors.YELLOW}⚠ MIP_MODEL_ENDPOINT not configured - skipping test{Colors.END}"
        )
        return None

    try:
        # Find test data
        data_root = settings.DATA_ROOT
        input_folder = os.path.join(data_root, "segmentation-examples")
        test_image = os.path.join(input_folder, "covid_1585.png")

        if not os.path.exists(test_image):
            print(f"{Colors.YELLOW}⚠ Test data not found at {test_image}{Colors.END}")
            print(
                f"{Colors.GREEN}✓ Skipping functional test (no test data){Colors.END}"
            )
            return True

        print(
            f"{Colors.GREEN}✓ Found test image: {os.path.basename(test_image)}{Colors.END}"
        )

        # Test the endpoint
        text_prompt = "left lung & right lung & COVID-19 infection"
        num_masks = len(text_prompt.split("&"))

        client = MedImageParseClient()
        response = client.submit(image_list=[test_image], prompts=[text_prompt])

        if not all(key in response[0] for key in ["image_features", "text_features"]):
            print(f"{Colors.RED}✗ Response does not contain expected keys{Colors.END}")
            return False

        if response[0]["image_features"].shape[0] != num_masks:
            print(
                f"{Colors.RED}✗ Expected {num_masks} masks, but got {response[0]['image_features'].shape[0]} masks{Colors.END}"
            )
            return False

        print(f"{Colors.GREEN}✓ Endpoint responded with expected format{Colors.END}")

        if not quiet:
            pretty_print_response(response)

        return True

    except Exception as e:
        print(f"{Colors.RED}✗ Endpoint test failed: {str(e)}{Colors.END}")
        traceback.print_exc()
        return False


def test_cxrreportgen_endpoint(quiet: bool = False) -> Optional[bool]:
    """Test CXRReportGen endpoint connectivity."""
    print(f"\n{Colors.BLUE}Testing CXRReportGen endpoint...{Colors.END}")

    # Check if endpoint is configured
    if not settings.CXRREPORTGEN_MODEL_ENDPOINT:
        print(
            f"{Colors.YELLOW}⚠ CXRREPORTGEN_MODEL_ENDPOINT not configured - skipping test{Colors.END}"
        )
        return None

    try:
        # Find test data
        data_root = settings.DATA_ROOT
        input_folder = os.path.join(data_root, "cxrreportgen-images")
        frontal = os.path.join(input_folder, "cxr_frontal.jpg")
        lateral = os.path.join(input_folder, "cxr_lateral.jpg")

        if not (os.path.exists(frontal) and os.path.exists(lateral)):
            print(f"{Colors.YELLOW}⚠ Test data not found at {input_folder}{Colors.END}")
            print(
                f"{Colors.GREEN}✓ Skipping functional test (no test data){Colors.END}"
            )
            return True

        print(
            f"{Colors.GREEN}✓ Found test images: {os.path.basename(frontal)}, {os.path.basename(lateral)}{Colors.END}"
        )

        # Test the endpoint
        indication = ""
        technique = ""
        comparison = "None"

        client = CxrReportGenClient()
        response = client.submit(
            frontal_image=frontal,
            lateral_image=lateral,
            indication=indication,
            technique=technique,
            comparison=comparison,
        )

        if not all(key in response[0] for key in ["output"]):
            print(f"{Colors.RED}✗ Response does not contain expected keys{Colors.END}")
            return False

        print(f"{Colors.GREEN}✓ Endpoint responded with expected format{Colors.END}")

        if not quiet:
            for i, r in enumerate(response):
                print(f"  == Response {i} ==")
                output = r["output"]
                print(f"  output:")
                if output is not None:
                    for row in output:
                        print(f"    {row}")

        return True

    except Exception as e:
        print(f"{Colors.RED}✗ Endpoint test failed: {str(e)}{Colors.END}")
        traceback.print_exc()
        return False


def test_gigapath_endpoint(quiet: bool = False) -> Optional[bool]:
    """Test GigaPath endpoint connectivity."""
    print(f"\n{Colors.BLUE}Testing GigaPath endpoint...{Colors.END}")

    # Check if endpoint is configured
    if not settings.GIGAPATH_MODEL_ENDPOINT:
        print(
            f"{Colors.YELLOW}⚠ GIGAPATH_MODEL_ENDPOINT not configured - skipping test{Colors.END}"
        )
        return None

    try:
        # Find test data
        data_root = settings.DATA_ROOT
        input_folder = os.path.join(
            data_root, "advanced-radpath-demo", "sample_images", "pathology"
        )
        test_image = os.path.join(input_folder, "TCGA-19-2631.png")

        if not os.path.exists(test_image):
            print(f"{Colors.YELLOW}⚠ Test data not found at {test_image}{Colors.END}")
            print(
                f"{Colors.GREEN}✓ Skipping functional test (no test data){Colors.END}"
            )
            return True

        print(
            f"{Colors.GREEN}✓ Found test image: {os.path.basename(test_image)}{Colors.END}"
        )

        # Test the endpoint
        client = GigaPathClient()
        response = client.submit(image_list=[test_image])

        print(f"{Colors.GREEN}✓ Endpoint responded successfully{Colors.END}")

        if not quiet:
            pretty_print_response(response)

        return True

    except Exception as e:
        print(f"{Colors.RED}✗ Endpoint test failed: {str(e)}{Colors.END}")
        traceback.print_exc()
        return False


def test_gpt_endpoint(quiet: bool = False) -> Optional[bool]:
    """Test GPT endpoint connectivity (optional)."""

    # Check if endpoint is configured
    if not settings.AZURE_OPENAI_ENDPOINT:
        print(
            f"{Colors.YELLOW}⚠ AZURE_OPENAI_ENDPOINT not configured - skipping test{Colors.END}"
        )
        return None

    try:
        print(f"\n{Colors.BLUE}Testing GPT endpoint...{Colors.END}")
        # Check if API key is also available
        if not settings.AZURE_OPENAI_API_KEY:
            print(f"{Colors.RED}⚠ AZURE_OPENAI_API_KEY not configured!{Colors.END}")
            return False

        if not settings.AZURE_OPENAI_MODEL_NAME:
            print(f"{Colors.RED}⚠ AZURE_OPENAI_MODEL_NAME not configured!{Colors.END}")
            return False

        print(f"{Colors.GREEN}✓ Creating OpenAI client...{Colors.END}")
        client = create_openai_client()

        # Simple test - get available models
        print(f"{Colors.GREEN}✓ Testing basic connectivity...{Colors.END}")

        # Try a simple completion request
        response = client.chat.completions.create(
            model=settings.AZURE_OPENAI_MODEL_NAME,  # This should be the deployed model name
            messages=[
                {
                    "role": "user",
                    "content": "Hello! Can you respond with just 'Hello from GPT'?",
                }
            ],
            max_tokens=100,
            temperature=0.0,
        )

        if response and response.choices:
            response_text = response.choices[0].message.content
            print(f"{Colors.GREEN}✓ GPT responded: {response_text}{Colors.END}")

            if not quiet:
                print(f"  Model: {response.model}")
                print(f"  Usage: {response.usage}")

            return True
        else:
            print(f"{Colors.RED}✗ GPT response was empty or invalid{Colors.END}")
            return False

    except Exception as e:
        print(f"{Colors.RED}✗ GPT endpoint test failed: {str(e)}{Colors.END}")
        return False


def print_configuration():
    """Print current configuration values from settings."""
    print(f"{Colors.CYAN}Configuration{Colors.END}")
    print("=" * 40)

    # Print endpoint configurations
    print(f"\n{Colors.BLUE}Model Endpoints:{Colors.END}")
    print(
        f"  MI2_MODEL_ENDPOINT:\n    {settings.MI2_MODEL_ENDPOINT or f'{Colors.YELLOW}(not set){Colors.END}'}"
    )
    print(
        f"  MIP_MODEL_ENDPOINT:\n    {settings.MIP_MODEL_ENDPOINT or f'{Colors.YELLOW}(not set){Colors.END}'}"
    )
    print(
        f"  GIGAPATH_MODEL_ENDPOINT:\n    {settings.GIGAPATH_MODEL_ENDPOINT or f'{Colors.YELLOW}(not set){Colors.END}'}"
    )
    print(
        f"  CXRREPORTGEN_MODEL_ENDPOINT:\n    {settings.CXRREPORTGEN_MODEL_ENDPOINT or f'{Colors.YELLOW}(not set){Colors.END}'}"
    )

    # Print Azure OpenAI configuration
    print(f"\n{Colors.PURPLE}Azure OpenAI Configuration:{Colors.END}")
    print(
        f"  AZURE_OPENAI_ENDPOINT:\n    {settings.AZURE_OPENAI_ENDPOINT or f'{Colors.YELLOW}(not set){Colors.END}'}"
    )

    if settings.AZURE_OPENAI_ENDPOINT:
        print(
            f"  AZURE_OPENAI_MODEL_NAME:\n    {settings.AZURE_OPENAI_MODEL_NAME or f'{Colors.YELLOW}(not set){Colors.END}'}"
        )
        if settings.AZURE_OPENAI_API_KEY:
            # Mask the API key for security
            masked_key = (
                settings.AZURE_OPENAI_API_KEY[:8]
                + "*" * (len(settings.AZURE_OPENAI_API_KEY) - 16)
                + settings.AZURE_OPENAI_API_KEY[-8:]
                if len(settings.AZURE_OPENAI_API_KEY) > 16
                else "***HIDDEN***"
            )
            print(f"  AZURE_OPENAI_API_KEY:\n    {masked_key}")

    # Print data configuration
    print(f"\n{Colors.GREEN}Data Configuration:{Colors.END}")
    print(f"  DATA_ROOT: {settings.DATA_ROOT}")

    print()  # Empty line for readability


def main():
    """Main entry point for the test command."""
    parser = argparse.ArgumentParser(
        description="Test healthcare AI model endpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  healthcareai-test                        # Test all endpoints with detailed output
  healthcareai-test --models mi2           # Test only MedImageInsight
  healthcareai-test --models mi2,mip       # Test MedImageInsight and MedImageParse
  healthcareai-test --quiet                # Test with minimal output
        """,
    )

    parser.add_argument(
        "--models",
        type=str,
        help="Comma-delimited list of model endpoints to test. Options: mi2,mip,cxr,gpt,pgp. If not specified, tests all models.",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress detailed response output and use minimal logging",
    )

    args = parser.parse_args()

    # Parse models list
    if args.models:
        # Split by comma and strip whitespace
        selected_models = [model.strip().lower() for model in args.models.split(",")]
        # Validate model names
        valid_models = {"mi2", "mip", "cxr", "gpt", "pgp"}
        invalid_models = set(selected_models) - valid_models
        if invalid_models:
            print(
                f"{Colors.RED}Error: Invalid model(s): {', '.join(invalid_models)}{Colors.END}"
            )
            print(f"Valid models are: {', '.join(sorted(valid_models))}")
            sys.exit(1)
    else:
        # Default to all models if none specified
        selected_models = ["mi2", "mip", "cxr", "gpt", "pgp"]

    print(f"{Colors.BOLD}Healthcare AI Endpoint Tester{Colors.END}")
    print("=" * 40)

    # Show configuration if requested
    print_configuration()

    # Run tests based on selected models
    test_results = {}

    if "mi2" in selected_models:
        test_results["mi2"] = test_medimageinsight_endpoint(args.quiet)

    if "mip" in selected_models:
        test_results["mip"] = test_medimageparse_endpoint(args.quiet)

    if "cxr" in selected_models:
        test_results["cxr"] = test_cxrreportgen_endpoint(args.quiet)

    if "pgp" in selected_models:
        test_results["pgp"] = test_gigapath_endpoint(args.quiet)

    if "gpt" in selected_models:
        test_results["gpt"] = test_gpt_endpoint(args.quiet)

    # Summary
    if test_results:  # Only show summary if tests were run
        print("\n" + "=" * 40)
        print(f"{Colors.BOLD}Test Summary:{Colors.END}")

        passed = sum(1 for result in test_results.values() if result is True)
        failed = sum(1 for result in test_results.values() if result is False)
        skipped = sum(1 for result in test_results.values() if result is None)
        total = len(test_results)

        for model, result in test_results.items():
            if result is True:
                status = f"{Colors.GREEN}✓ PASS{Colors.END}"
            elif result is False:
                status = f"{Colors.RED}✗ FAIL{Colors.END}"
            else:  # result is None
                status = f"{Colors.YELLOW}- SKIP{Colors.END}"
            print(f"  {model.upper()}: {status}")

        print(
            f"\nOverall: {passed} passed, {failed} failed, {skipped} skipped ({total} total)"
        )

        if failed == 0:
            print(f"{Colors.GREEN}All configured endpoint tests passed!{Colors.END}")
            sys.exit(0)
        else:
            print(
                f"{Colors.YELLOW}Some endpoint tests failed. Check your configuration.{Colors.END}"
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
