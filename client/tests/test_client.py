"""
test_client.py — Unit tests for DengueNowcastClient.

Tests use unittest.mock to stub requests.Session.get; no live network calls.
Run with:  pytest client/tests/
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from dengue_nowcast_client import (
    DengueNowcastClient,
    DengueNowcastConnectionError,
    DengueNowcastHTTPError,
    DengueNowcastResponseError,
)

BASE = "http://testserver"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client() -> DengueNowcastClient:
    return DengueNowcastClient(base_url=BASE, timeout=5)


def _mock_ok(payload: object, status: int = 200) -> MagicMock:
    """Build a mock response that returns payload as JSON."""
    m = MagicMock()
    m.ok = status < 400
    m.status_code = status
    m.json.return_value = payload
    m.text = str(payload)
    return m


def _mock_error(status: int, detail: str = "error") -> MagicMock:
    m = MagicMock()
    m.ok = False
    m.status_code = status
    m.json.return_value = {"detail": detail}
    m.text = detail
    return m


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_returns_dict(client: DengueNowcastClient) -> None:
    payload = {"status": "ok", "store_loaded": True, "n_predictions": 26, "n_splits": 13}
    with patch.object(client._session, "get", return_value=_mock_ok(payload)) as mock_get:
        result = client.health()
        mock_get.assert_called_once_with(f"{BASE}/health", params=None, timeout=5)
    assert result["status"] == "ok"
    assert result["n_predictions"] == 26


# ---------------------------------------------------------------------------
# /predictions — parameter passing
# ---------------------------------------------------------------------------

def test_predictions_no_filters_passes_only_limit_offset(client: DengueNowcastClient) -> None:
    with patch.object(client._session, "get", return_value=_mock_ok({"count": 0, "items": []})) as mock_get:
        client.predictions()
    _, kwargs = mock_get.call_args
    params = kwargs["params"]
    # None-valued keys must be stripped
    assert "split_month" not in params
    assert "target_month" not in params
    assert "horizon" not in params
    # Default pagination always present
    assert params["limit"] == 200
    assert params["offset"] == 0


def test_predictions_all_filters(client: DengueNowcastClient) -> None:
    with patch.object(client._session, "get", return_value=_mock_ok({"count": 1, "items": []})) as mock_get:
        client.predictions(split_month="2025-06", target_month="2025-07", horizon=1, limit=10, offset=5)
    _, kwargs = mock_get.call_args
    params = kwargs["params"]
    assert params["split_month"] == "2025-06"
    assert params["target_month"] == "2025-07"
    assert params["horizon"] == 1
    assert params["limit"] == 10
    assert params["offset"] == 5


def test_predictions_url(client: DengueNowcastClient) -> None:
    with patch.object(client._session, "get", return_value=_mock_ok({"count": 0, "items": []})) as mock_get:
        client.predictions()
    url, _ = mock_get.call_args[0], mock_get.call_args[1]
    assert mock_get.call_args[0][0] == f"{BASE}/predictions"


# ---------------------------------------------------------------------------
# /predictions/by-target
# ---------------------------------------------------------------------------

def test_predictions_by_target_sends_target_month(client: DengueNowcastClient) -> None:
    with patch.object(client._session, "get", return_value=_mock_ok({"count": 2, "items": []})) as mock_get:
        result = client.predictions_by_target("2026-01")
    _, kwargs = mock_get.call_args
    assert kwargs["params"]["target_month"] == "2026-01"
    assert mock_get.call_args[0][0] == f"{BASE}/predictions/by-target"


def test_predictions_by_target_returns_count(client: DengueNowcastClient) -> None:
    with patch.object(client._session, "get", return_value=_mock_ok({"count": 3, "items": []})):
        result = client.predictions_by_target("2026-01")
    assert result["count"] == 3


# ---------------------------------------------------------------------------
# /evaluation
# ---------------------------------------------------------------------------

def test_evaluation_horizon_filter(client: DengueNowcastClient) -> None:
    with patch.object(client._session, "get", return_value=_mock_ok({"count": 0, "items": []})) as mock_get:
        client.evaluation(horizon=2)
    _, kwargs = mock_get.call_args
    assert kwargs["params"]["horizon"] == 2
    assert "split_month" not in kwargs["params"]


def test_evaluation_no_filter_strips_nones(client: DengueNowcastClient) -> None:
    with patch.object(client._session, "get", return_value=_mock_ok({"count": 0, "items": []})) as mock_get:
        client.evaluation()
    _, kwargs = mock_get.call_args
    # Both optional params are None → both stripped → params=None
    assert kwargs["params"] is None


# ---------------------------------------------------------------------------
# HTTP error handling
# ---------------------------------------------------------------------------

def test_http_404_raises_http_error(client: DengueNowcastClient) -> None:
    with patch.object(client._session, "get", return_value=_mock_error(404, "not found")):
        with pytest.raises(DengueNowcastHTTPError) as exc_info:
            client.splits()
    assert exc_info.value.status_code == 404
    assert "404" in str(exc_info.value)


def test_http_503_raises_http_error(client: DengueNowcastClient) -> None:
    with patch.object(client._session, "get", return_value=_mock_error(503, "service unavailable")):
        with pytest.raises(DengueNowcastHTTPError) as exc_info:
            client.health()
    assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# Network error handling
# ---------------------------------------------------------------------------

def test_timeout_raises_connection_error(client: DengueNowcastClient) -> None:
    with patch.object(client._session, "get", side_effect=requests.exceptions.Timeout()):
        with pytest.raises(DengueNowcastConnectionError) as exc_info:
            client.health()
    assert "timed out" in str(exc_info.value).lower()


def test_connection_refused_raises_connection_error(client: DengueNowcastClient) -> None:
    with patch.object(client._session, "get", side_effect=requests.exceptions.ConnectionError()):
        with pytest.raises(DengueNowcastConnectionError):
            client.splits()


def test_generic_request_exception_raises_connection_error(client: DengueNowcastClient) -> None:
    with patch.object(client._session, "get", side_effect=requests.exceptions.RequestException("boom")):
        with pytest.raises(DengueNowcastConnectionError):
            client.predictions()


# ---------------------------------------------------------------------------
# JSON parse error
# ---------------------------------------------------------------------------

def test_bad_json_raises_response_error(client: DengueNowcastClient) -> None:
    bad = MagicMock()
    bad.ok = True
    bad.status_code = 200
    bad.json.side_effect = ValueError("not json")
    with patch.object(client._session, "get", return_value=bad):
        with pytest.raises(DengueNowcastResponseError):
            client.health()


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

def test_context_manager_closes_session(client: DengueNowcastClient) -> None:
    with patch.object(client._session, "close") as mock_close:
        with client:
            pass
        mock_close.assert_called_once()


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

def test_repr(client: DengueNowcastClient) -> None:
    r = repr(client)
    assert "testserver" in r
    assert "timeout=5" in r
