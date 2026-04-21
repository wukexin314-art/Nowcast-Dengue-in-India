"""
client.py — DengueNowcastClient: Python wrapper for the Dengue Nowcast History API.
"""
from __future__ import annotations

from typing import Any

import requests

from .exceptions import (
    DengueNowcastConnectionError,
    DengueNowcastHTTPError,
    DengueNowcastResponseError,
)

DEFAULT_BASE_URL = "https://dengue-nowcast-api.onrender.com"
DEFAULT_TIMEOUT = 30


class DengueNowcastClient:
    """
    Client for the India Dengue Nowcast History API.

    Parameters
    ----------
    base_url:
        Root URL of the deployed API.  Defaults to the production service at
        ``https://dengue-nowcast-api.onrender.com``.
        Override for local development: ``DengueNowcastClient(base_url="http://localhost:8001")``.
    timeout:
        Seconds to wait for a response before raising
        :class:`~dengue_nowcast_client.DengueNowcastConnectionError`.

    Examples
    --------
    >>> client = DengueNowcastClient()
    >>> client.health()
    {'status': 'ok', 'store_loaded': True, ...}
    >>> client.predictions(split_month="2025-06", horizon=1)
    {'count': 2, 'items': [...]}
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Send a GET request; strip None-valued params; raise on error."""
        url = f"{self.base_url}{path}"
        clean: dict[str, Any] = {
            k: v for k, v in (params or {}).items() if v is not None
        }
        try:
            response = self._session.get(
                url,
                params=clean or None,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout as exc:
            raise DengueNowcastConnectionError(
                f"Request timed out after {self.timeout}s — {url}"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise DengueNowcastConnectionError(
                f"Connection failed — {url}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise DengueNowcastConnectionError(
                f"Request error: {exc}"
            ) from exc

        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise DengueNowcastHTTPError(response.status_code, str(detail))

        try:
            return response.json()
        except ValueError as exc:
            raise DengueNowcastResponseError(
                f"Could not parse JSON from {url}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """
        GET /health — service status and prediction-store statistics.

        Returns
        -------
        dict with keys: status, store_loaded, n_predictions, n_splits,
        split_month_min, split_month_max, built_at.

        Raises
        ------
        DengueNowcastConnectionError
            Network-level failure.
        DengueNowcastHTTPError
            API returned a non-2xx status.
        """
        return self._get("/health")

    def splits(self) -> dict[str, Any]:
        """
        GET /splits — all available split months in chronological order.

        Returns
        -------
        dict with keys: count (int), items (list[str] of YYYY-MM strings).
        """
        return self._get("/splits")

    def predictions(
        self,
        split_month: str | None = None,
        target_month: str | None = None,
        horizon: int | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        GET /predictions — historical prediction records.

        All filter parameters are optional.  Omitting one means "no restriction".
        Results are sorted split_month → target_month → horizon.

        Parameters
        ----------
        split_month:
            Restrict to one rolling-split (e.g. ``"2025-06"``).
        target_month:
            Restrict to predictions for one target month (e.g. ``"2025-08"``).
        horizon:
            Restrict to a specific forecast step (``1`` = one month ahead,
            ``2`` = two months ahead).
        limit:
            Maximum number of rows to return (1–1000).  Default 200.
        offset:
            Row offset for pagination.  Default 0.

        Returns
        -------
        dict with keys: count (int), items (list[PredictionRecord]).
        """
        return self._get("/predictions", {
            "split_month":  split_month,
            "target_month": target_month,
            "horizon":      horizon,
            "limit":        limit,
            "offset":       offset,
        })

    def predictions_by_target(self, target_month: str) -> dict[str, Any]:
        """
        GET /predictions/by-target — full prediction history for one target month.

        Returns every rolling split's forecast for ``target_month``, sorted by
        split_month.  Useful for seeing how the nowcast evolved as more data arrived.

        Parameters
        ----------
        target_month:
            Target month to query, e.g. ``"2026-03"``.

        Returns
        -------
        dict with keys: count (int), items (list[PredictionRecord]).
        """
        return self._get("/predictions/by-target", {"target_month": target_month})

    def evaluation(
        self,
        split_month: str | None = None,
        horizon: int | None = None,
    ) -> dict[str, Any]:
        """
        GET /evaluation — rolling-split evaluation metrics (RMSE, MAPE, etc.).

        Parameters
        ----------
        split_month:
            Filter to one split (e.g. ``"2025-09"``).
        horizon:
            Filter by ``test_horizon_months`` (e.g. ``2`` for a 2-month test window).

        Returns
        -------
        dict with keys: count (int), items (list[EvaluationRecord]).

        Note: ``MAPE_train_%``, ``MAPE_test_%``, and
        ``MAPE_test_seasonal_naive_%`` from the raw CSV are returned by the API
        as ``MAPE_train_pct``, ``MAPE_test_pct``, and
        ``MAPE_test_seasonal_naive_pct``.
        """
        return self._get("/evaluation", {
            "split_month": split_month,
            "horizon":     horizon,
        })

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying requests.Session."""
        self._session.close()

    def __enter__(self) -> "DengueNowcastClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"DengueNowcastClient(base_url={self.base_url!r}, timeout={self.timeout})"
