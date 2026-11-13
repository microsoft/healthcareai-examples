"""Check or wait for a parent Azure ML pipeline, including scoring and evaluation."""

import argparse
import time
from pathlib import Path

from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential

TERMINAL_STATUSES = {"Completed", "Failed", "Canceled", "NotResponding"}


def check_job(
    ml_client: MLClient,
    job_name: str,
    *,
    status_only: bool = False,
    wait_for: str | None = None,
    timeout_minutes: float = 360,
    poll_seconds: float = 30,
) -> int:
    """Print status changes; return 0 for success, 1 for failure, 2 for timeout."""
    if timeout_minutes <= 0 or poll_seconds <= 0:
        raise ValueError("Timeout and poll interval must be positive.")
    deadline = time.monotonic() + timeout_minutes * 60
    previous_status = None
    target_status = wait_for.casefold() if wait_for else None
    while True:
        job = ml_client.jobs.get(job_name)
        status = str(getattr(job.status, "value", job.status))
        if status != previous_status:
            print(status, flush=True)
            previous_status = status
        if status_only:
            return 0
        if target_status is not None and status.casefold() == target_status:
            return 0
        if status in TERMINAL_STATUSES:
            return 0 if target_status is None and status == "Completed" else 1
        if time.monotonic() >= deadline:
            print(f"Timed out waiting for {wait_for or 'completion'}.")
            return 2
        time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "job_name", help="Parent pipeline name, not the training child."
    )
    parser.add_argument("--workspace-config", type=Path, default=Path("config.json"))
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current status and exit without waiting.",
    )
    parser.add_argument(
        "--wait-for", help="Wait for a particular state instead of completion."
    )
    parser.add_argument("--timeout-minutes", type=float, default=360)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.timeout_minutes <= 0 or args.poll_seconds <= 0:
        parser.error("Timeout and poll interval must be positive.")
    ml_client = MLClient.from_config(AzureCliCredential(), path=args.workspace_config)
    return check_job(
        ml_client,
        args.job_name,
        status_only=args.status,
        wait_for=args.wait_for,
        timeout_minutes=args.timeout_minutes,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
