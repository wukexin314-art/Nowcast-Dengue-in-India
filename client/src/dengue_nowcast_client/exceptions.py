"""
exceptions.py — Custom exception hierarchy for dengue-nowcast-client.
"""
from __future__ import annotations


class DengueNowcastError(Exception):
    """Base class for all dengue-nowcast-client errors."""


class DengueNowcastHTTPError(DengueNowcastError):
    """The API returned a non-2xx HTTP status code."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class DengueNowcastConnectionError(DengueNowcastError):
    """Network-level failure: timeout, DNS error, connection refused, etc."""


class DengueNowcastResponseError(DengueNowcastError):
    """The response body could not be parsed or was structurally unexpected."""
