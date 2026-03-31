"""
fetch_who.py — Fetch WHO monthly dengue cases for India.

Implements a three-layer fallback strategy:
  Layer 1 (shiny)    — Connect to the WHO Shiny dashboard via WebSocket,
                       establish a session, accept terms, and download the
                       Excel file through the session's download handler.
                       No authentication required.
  Layer 2 (download) — Direct file download link on the WHO dashboard page.
                       Set config.WHO_DOWNLOAD_URL to activate.
  Layer 3 (html)     — Parse HTML from the WHO Shiny dashboard (fragile,
                       last resort).

Returns a DataFrame with columns:
    date                 (str, YYYY-MM)
    who_cases_monthly    (float)

Returns None on any unrecoverable error; existing stored values are NOT
overwritten in that case.

Coverage note:
    India's WHO monthly dengue data starts at 2024-01.  Months 2021-01
    through 2023-12 will be absent from the fetch result and will remain
    null in monthly_data.csv.  This is a source limitation — India began
    monthly reporting to WHO from January 2024.  It is NOT a fetch failure
    or a code bug.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)

_INDIA_ALIASES = {"india", "ind", "in"}

# Shiny app constants
_SHINY_BASE = "https://worldhealthorg.shinyapps.io/dengue_global/"
_SHINY_WS_TIMEOUT = 30  # seconds to wait for each WebSocket message


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fetch_who() -> tuple[Optional[pd.DataFrame], str]:
    """
    Fetch WHO monthly dengue cases for India using a layered fallback strategy.

    Returns:
        (df, fetch_mode) where df is a DataFrame(date, who_cases_monthly)
        or None on failure, and fetch_mode is 'shiny'|'download'|'html'|'failed'.
    """
    # Layer 1: Shiny WebSocket + download handler
    df, mode = _try_shiny_download()
    if df is not None:
        return df, mode

    # Layer 2: direct download link (if manually configured)
    if config.WHO_DOWNLOAD_URL:
        df, mode = _try_direct_url(config.WHO_DOWNLOAD_URL, mode="download")
        if df is not None:
            return df, mode

    # Layer 3: HTML parsing (fragile)
    logger.warning(
        "fetch_who | Shiny download failed and no WHO_DOWNLOAD_URL configured. "
        "Attempting HTML fallback — this is fragile and may break without warning."
    )
    df, mode = _try_html_fallback()
    if df is not None:
        return df, mode

    logger.error(
        "fetch_who | All layers failed.  Existing stored values will be preserved.\n"
        "To fix: inspect %s and set config.WHO_DOWNLOAD_URL.",
        config.WHO_DASHBOARD_URL,
    )
    return None, "failed"


# ---------------------------------------------------------------------------
# Layer 1: Shiny WebSocket download
# ---------------------------------------------------------------------------

def _try_shiny_download() -> tuple[Optional[pd.DataFrame], str]:
    """
    Connect to the WHO Shiny dashboard via WebSocket, init a session,
    accept the terms modal, and download the Excel file while the session
    is still active.
    """
    logger.info("fetch_who | Trying Shiny WebSocket download layer")
    try:
        raw_bytes = asyncio.run(_shiny_download_async())
    except Exception as exc:
        logger.warning("fetch_who | Shiny download failed: %s", exc)
        return None, "shiny"

    if raw_bytes is None:
        return None, "shiny"

    # Parse the downloaded Excel file
    try:
        df = _parse_who_excel(raw_bytes)
    except Exception as exc:
        logger.warning("fetch_who | Shiny download parse failed: %s", exc)
        return None, "shiny"

    if df is None or df.empty:
        logger.warning("fetch_who | Shiny download returned empty result after parsing")
        return None, "shiny"

    # Save raw response
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = config.RAW_WHO / f"who_shiny_{ts}.xlsx"
    raw_path.write_bytes(raw_bytes)
    logger.info("fetch_who | Raw response saved to %s", raw_path)

    logger.info("fetch_who | Shiny layer succeeded: %d rows", len(df))
    return df, "shiny"


async def _shiny_download_async() -> Optional[bytes]:
    """
    Async implementation of the Shiny WebSocket download flow:
      1. GET dashboard HTML → extract worker prefix
      2. Connect WebSocket → receive sessionId
      3. Send init (with download outputs visible) → receive download URL
      4. Accept terms modal (send closeModal click)
      5. Download Excel file via HTTP while WebSocket stays open
    """
    try:
        import websockets
    except ImportError:
        logger.warning(
            "fetch_who | Shiny layer requires 'websockets' — not installed. "
            "Run: pip install websockets"
        )
        return None

    # Step 1: GET the dashboard HTML to find the worker prefix
    logger.info("fetch_who | Fetching dashboard HTML to discover worker ID")
    http_session = requests.Session()
    try:
        resp = http_session.get(_SHINY_BASE, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("fetch_who | Failed to fetch dashboard HTML: %s", exc)
        return None

    # Extract worker prefix (pattern: _w_<32-char hex>)
    worker_match = re.search(r'(_w_[0-9a-f]{32})', resp.text)
    if not worker_match:
        logger.warning("fetch_who | Could not find worker prefix in dashboard HTML")
        return None

    worker_prefix = worker_match.group(1)
    logger.info("fetch_who | Found worker prefix: %s", worker_prefix)

    # Step 2: Connect via WebSocket
    ws_url = f"wss://worldhealthorg.shinyapps.io/dengue_global/{worker_prefix}/websocket/"
    ws_headers = {"Origin": "https://worldhealthorg.shinyapps.io"}
    cookie_header = "; ".join([f"{k}={v}" for k, v in http_session.cookies.items()])
    if cookie_header:
        ws_headers["Cookie"] = cookie_header

    logger.info("fetch_who | Connecting WebSocket: %s", ws_url)

    async with websockets.connect(
        ws_url,
        additional_headers=ws_headers,
        open_timeout=_SHINY_WS_TIMEOUT,
        close_timeout=5,
    ) as ws:
        # Receive config message with sessionId
        raw_msg = await asyncio.wait_for(ws.recv(), timeout=_SHINY_WS_TIMEOUT)
        json_match = re.search(r"\{.*\}", str(raw_msg), re.DOTALL)
        if not json_match:
            logger.warning("fetch_who | Unexpected first WS message: %s", str(raw_msg)[:200])
            return None

        config_msg = json.loads(json_match.group())
        if "config" not in config_msg:
            logger.warning("fetch_who | No 'config' in first WS message")
            return None

        session_id = config_msg["config"]["sessionId"]
        logger.info("fetch_who | Got sessionId: %s", session_id)

        # Step 3: Send init — mark download-related outputs as visible
        init_data = {
            "method": "init",
            "data": {
                ".clientdata_output_dl_all_data_hidden": False,
                ".clientdata_output_dl_data_country_hidden": False,
                ".clientdata_output_dl_data_region_hidden": False,
                ".clientdata_output_geo_filter_hidden": False,
                ".clientdata_output_dt_raw_data_hidden": False,
                ".clientdata_pixelratio": 2,
                ".clientdata_url_protocol": "https:",
                ".clientdata_url_hostname": "worldhealthorg.shinyapps.io",
                ".clientdata_url_port": "",
                ".clientdata_url_pathname": f"/dengue_global/{worker_prefix}/",
                ".clientdata_url_search": "",
                ".clientdata_url_hash_initial": "",
                ".clientdata_singletons": "",
                ".clientdata_allowDataUriScheme": True,
            },
        }
        await ws.send(json.dumps(init_data))
        logger.info("fetch_who | Sent init message, waiting for download URL...")

        # Wait for the response containing the download URL
        download_url = None
        for _ in range(30):
            try:
                raw_msg = await asyncio.wait_for(ws.recv(), timeout=_SHINY_WS_TIMEOUT)
            except asyncio.TimeoutError:
                break

            json_match = re.search(r"\{.*\}", str(raw_msg), re.DOTALL)
            if not json_match:
                continue
            try:
                payload = json.loads(json_match.group())
            except json.JSONDecodeError:
                continue

            values = payload.get("values", {})
            if "dl_all_data" in values:
                download_url = values["dl_all_data"]
                logger.info("fetch_who | Got download path: %s", download_url)
                break

        if not download_url:
            logger.warning("fetch_who | Never received dl_all_data download URL")
            return None

        # Step 4: Accept terms modal
        logger.info("fetch_who | Accepting terms modal (closeModal)...")
        await ws.send(json.dumps({"method": "update", "data": {"closeModal": 1}}))
        await asyncio.sleep(2)

        # Drain modal-close responses
        for _ in range(10):
            try:
                await asyncio.wait_for(ws.recv(), timeout=2)
            except asyncio.TimeoutError:
                break

        # Step 5: Navigate to the download tab and set the geographic filter.
        # The Shiny app uses tabsetPanel ID "2709" for the main navbar;
        # the download sub-tab is "dl_data" under the "meta" parent tab.
        # The download handler also requires dl_filter_adm_level to be set.
        logger.info("fetch_who | Navigating to download tab...")

        # Find the tabset ID that contains dl_data from the HTML
        tabset_match = re.search(
            r'data-tabsetid="(\d+)"[^>]*>.*?data-value="dl_data"',
            resp.text, re.DOTALL,
        )
        main_tabset_id = tabset_match.group(1) if tabset_match else "2709"

        # Set main tab to "meta" (parent of dl_data), then sub-tab to "dl_data"
        await ws.send(json.dumps({
            "method": "update",
            "data": {main_tabset_id: "meta"},
        }))
        await asyncio.sleep(0.5)

        # Try all discovered tabset IDs for the dl_data sub-tab
        tabset_ids = set(re.findall(r'data-tabsetid="(\d+)"', resp.text))
        for tid in tabset_ids:
            if tid != main_tabset_id:
                await ws.send(json.dumps({
                    "method": "update",
                    "data": {tid: "dl_data"},
                }))
        await asyncio.sleep(1)

        # Set geographic filter to "global" (download all data)
        await ws.send(json.dumps({
            "method": "update",
            "data": {"dl_filter_adm_level": "global"},
        }))
        await asyncio.sleep(2)

        # Drain navigation responses and check for updated download URL
        for _ in range(20):
            try:
                raw_msg = await asyncio.wait_for(ws.recv(), timeout=5)
                json_match = re.search(r"\{.*\}", str(raw_msg), re.DOTALL)
                if json_match:
                    payload = json.loads(json_match.group())
                    if "values" in payload and "dl_all_data" in payload["values"]:
                        download_url = payload["values"]["dl_all_data"]
            except (asyncio.TimeoutError, json.JSONDecodeError):
                break

        # Step 6: Download the Excel file while WebSocket stays open
        full_url = f"{_SHINY_BASE}{download_url}"
        logger.info("fetch_who | Downloading Excel: %s", full_url)

        try:
            dl_resp = http_session.get(full_url, timeout=120)
            logger.info(
                "fetch_who | Download response: status=%d, content-type=%s, size=%d bytes",
                dl_resp.status_code,
                dl_resp.headers.get("Content-Type", "unknown"),
                len(dl_resp.content),
            )
            dl_resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("fetch_who | Download request failed: %s", exc)
            return None

        if len(dl_resp.content) < 500:
            logger.warning(
                "fetch_who | Download too small (%d bytes), likely not a valid Excel file",
                len(dl_resp.content),
            )
            return None

        return dl_resp.content


# ---------------------------------------------------------------------------
# Layer 2: Direct URL download
# ---------------------------------------------------------------------------

def _try_direct_url(url: str, mode: str) -> tuple[Optional[pd.DataFrame], str]:
    """
    Attempt to download and parse WHO data from a direct URL.
    Supports CSV, Excel (.xlsx/.xls), and JSON responses.
    """
    logger.info("fetch_who | Trying %s layer: %s", mode, url)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("fetch_who | %s layer request failed: %s", mode, exc)
        return None, mode

    content_type = resp.headers.get("Content-Type", "").lower()
    raw_bytes = resp.content

    try:
        if "json" in content_type:
            df = _parse_who_json(resp.json())
        elif "excel" in content_type or url.endswith((".xlsx", ".xls")):
            df = _parse_who_excel(raw_bytes)
        else:
            df = _parse_who_csv(raw_bytes)
    except Exception as exc:
        logger.warning("fetch_who | %s layer parse failed: %s", mode, exc)
        return None, mode

    if df is None or df.empty:
        logger.warning("fetch_who | %s layer returned empty result after parsing", mode)
        return None, mode

    # Save raw response
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = "json" if "json" in content_type else ("xlsx" if "excel" in content_type else "csv")
    raw_path = config.RAW_WHO / f"who_{mode}_{ts}.{ext}"
    raw_path.write_bytes(raw_bytes)
    logger.info("fetch_who | Raw response saved to %s", raw_path)

    logger.info("fetch_who | %s layer succeeded: %d rows", mode, len(df))
    return df, mode


# ---------------------------------------------------------------------------
# Layer 3: HTML fallback
# ---------------------------------------------------------------------------

def _try_html_fallback() -> tuple[Optional[pd.DataFrame], str]:
    """
    FRAGILE: Attempt to extract India monthly cases from the WHO Shiny
    dashboard HTML.  This is a last-resort fallback.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning(
            "fetch_who | HTML fallback requires 'beautifulsoup4' — not installed. "
            "Run: pip install beautifulsoup4"
        )
        return None, "failed"

    logger.warning("fetch_who | Using HTML fallback — results may be incomplete or incorrect")
    try:
        resp = requests.get(config.WHO_DASHBOARD_URL, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        tables = soup.find_all("table")
        for table in tables:
            try:
                df_candidate = pd.read_html(str(table))[0]
                result = _extract_india_monthly(df_candidate)
                if result is not None:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    (config.RAW_WHO / f"who_html_{ts}.html").write_text(resp.text, encoding="utf-8")
                    logger.info("fetch_who | HTML fallback succeeded: %d rows", len(result))
                    return result, "html"
            except Exception:
                continue

        logger.warning("fetch_who | HTML fallback: no parseable India monthly table found")
        return None, "html"

    except Exception as exc:
        logger.warning("fetch_who | HTML fallback error: %s", exc)
        return None, "failed"


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_who_csv(raw_bytes: bytes) -> Optional[pd.DataFrame]:
    """Parse a raw CSV bytes payload and extract India monthly cases."""
    df = pd.read_csv(io.BytesIO(raw_bytes))
    return _extract_india_monthly(df)


def _parse_who_excel(raw_bytes: bytes) -> Optional[pd.DataFrame]:
    """Parse a raw Excel bytes payload and extract India monthly cases."""
    xls = pd.ExcelFile(io.BytesIO(raw_bytes))
    for sheet in xls.sheet_names:
        try:
            df = xls.parse(sheet)
            result = _extract_india_monthly(df)
            if result is not None and not result.empty:
                return result
        except Exception:
            continue
    return None


def _parse_who_json(data: dict | list) -> Optional[pd.DataFrame]:
    """Parse a WHO JSON response."""
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        for key in ("data", "records", "value", "rows"):
            if key in data and isinstance(data[key], list):
                df = pd.DataFrame(data[key])
                break
        else:
            df = pd.json_normalize(data)
    else:
        raise ValueError(f"Unexpected JSON type: {type(data)}")

    return _extract_india_monthly(df)


def _extract_india_monthly(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Given a raw DataFrame from any WHO source, locate the India monthly
    new-cases column, normalize dates, and return a clean two-column DataFrame.
    """
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # --- Identify the country/region column ---
    country_col = _find_column(df, ["country", "region", "adm0_name", "iso3", "country_code"])
    if country_col is None:
        logger.warning("fetch_who | Cannot identify country column in: %s", list(df.columns))
        return None

    # Filter to India rows
    india_mask = df[country_col].astype(str).str.strip().str.lower().isin(_INDIA_ALIASES)
    if not india_mask.any():
        india_mask = df[country_col].astype(str).str.lower().str.contains("india", na=False)
    if not india_mask.any():
        logger.warning("fetch_who | No India rows found in country column '%s'", country_col)
        return None
    df = df[india_mask].copy()

    # --- Identify the date column ---
    date_col = _find_column(df, ["date", "month", "year_month", "period", "report_date", "yearmonth"])
    if date_col is None:
        logger.warning("fetch_who | Cannot identify date column in: %s", list(df.columns))
        return None

    # --- Identify the cases column ---
    cases_col = _find_column(df, [
        "cases", "new_cases", "confirmed_cases", "dengue_cases",
        "monthly_cases", "count", "value", "dengue", "dengue_fever",
    ])
    if cases_col is None:
        logger.warning("fetch_who | Cannot identify cases column in: %s", list(df.columns))
        return None

    # Validate: values should be non-negative numbers
    try:
        case_values = pd.to_numeric(df[cases_col], errors="coerce")
        if case_values.isnull().all():
            logger.warning("fetch_who | Cases column '%s' has no numeric values", cases_col)
            return None
    except Exception as exc:
        logger.warning("fetch_who | Error validating cases column: %s", exc)
        return None

    # Normalize dates
    try:
        df["date"] = df[date_col].apply(config.normalize_monthly_date)
    except Exception as exc:
        logger.warning("fetch_who | Date normalization failed: %s", exc)
        return None

    result = pd.DataFrame({
        "date": df["date"],
        "who_cases_monthly": pd.to_numeric(df[cases_col], errors="coerce"),
    })
    result = result.dropna(subset=["date"])
    result = result[result["date"] >= config.START_DATE]
    result = result.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    if result.empty:
        logger.warning("fetch_who | No rows remain after filtering to date >= %s", config.START_DATE)
        return None

    logger.info(
        "fetch_who | Extracted %d India monthly rows (date range: %s → %s)",
        len(result), result["date"].min(), result["date"].max(),
    )
    return result


def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first column name in df.columns that matches any candidate."""
    cols_lower = {c: c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    return None
