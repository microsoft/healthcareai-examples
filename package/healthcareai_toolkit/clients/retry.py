# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging

import requests
from ratelimit import RateLimitException, limits, sleep_and_retry
from requests.exceptions import ConnectionError, HTTPError, Timeout
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)


def create_retry_post_func(
    func=None,
    retries=15,
    rate_calls=None,
    rate_period=None,
    exp_multiplier=1,
    exp_min=2,
    exp_max=120,
    logger=None,
):
    """Create a post function with retries and rate limiting."""

    def is_retryable_exception(exc):
        if isinstance(exc, (ConnectionError, Timeout, RateLimitException)):
            return True
        elif isinstance(exc, HTTPError) and exc.response is not None:
            if 500 <= exc.response.status_code < 600 or exc.response.status_code == 429:
                return True
        return False

    before_sleep = None
    if logger is not None:
        before_sleep = before_sleep_log(logger, logging.DEBUG)

    if func is None:

        def func(*args, **kwargs):
            response = requests.post(*args, **kwargs)
            response.raise_for_status()
            return response

    if rate_calls and rate_period:
        func = sleep_and_retry(limits(calls=rate_calls, period=rate_period)(func))

    retry_dec = retry(
        retry=retry_if_exception(is_retryable_exception),
        wait=wait_random_exponential(
            multiplier=exp_multiplier, min=exp_min, max=exp_max
        ),
        stop=stop_after_attempt(retries),
        before_sleep=before_sleep,
    )

    func = retry_dec(func)

    def post_with_retry(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RetryError:
            print(f"Failed after {retries} retries.")
            raise

    return post_with_retry


class RetryClient(object):
    _default_retry_params = {
        "retries": 15,
        "rate_calls": None,
        "rate_period": None,
        "exp_multiplier": 1,
        "exp_min": 0.5,
        "exp_max": 120,
    }

    def __init__(self, func=None, retry_params=None, logger=None):
        retry_params = {**self._default_retry_params, **(retry_params or {})}

        self.logger = logger
        self.request_handler = create_retry_post_func(
            func=func, logger=self.logger, **retry_params
        )

    def submit_payload(self, payload, target, headers):
        response = self.request_handler(target, json=payload, headers=headers)
        return response.json()
