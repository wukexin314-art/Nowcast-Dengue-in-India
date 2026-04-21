"""
dengue_nowcast_client — Python client for the India Dengue Nowcast History API.

Quick start::

    from dengue_nowcast_client import DengueNowcastClient

    client = DengueNowcastClient()
    print(client.health())
    print(client.splits())
    print(client.predictions(limit=5))
    print(client.predictions_by_target("2026-01"))
    print(client.evaluation(horizon=2))
"""
from .client import DengueNowcastClient
from .exceptions import (
    DengueNowcastConnectionError,
    DengueNowcastError,
    DengueNowcastHTTPError,
    DengueNowcastResponseError,
)

__all__ = [
    "DengueNowcastClient",
    "DengueNowcastError",
    "DengueNowcastHTTPError",
    "DengueNowcastConnectionError",
    "DengueNowcastResponseError",
]

__version__ = "0.1.0"
