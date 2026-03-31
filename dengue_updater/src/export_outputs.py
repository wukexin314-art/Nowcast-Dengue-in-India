"""
export_outputs.py — Export final master_data.xlsx with two sheets.

Sheet 1: monthly_data  (from monthly_data.csv)
Sheet 2: yearly_data   (from yearly_data.csv)
"""

from __future__ import annotations

import logging

import pandas as pd

import config

logger = logging.getLogger(__name__)


def export_master_xlsx() -> None:
    """
    Read the processed CSVs and write master_data.xlsx with two sheets.

    Does nothing (logs a warning) if a source CSV does not exist.
    """
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    sheets: dict[str, pd.DataFrame] = {}

    if config.MONTHLY_DATA_CSV.exists():
        sheets["monthly_data"] = pd.read_csv(config.MONTHLY_DATA_CSV)
    else:
        logger.warning(
            "export_outputs | %s not found — monthly_data sheet will be empty",
            config.MONTHLY_DATA_CSV,
        )
        sheets["monthly_data"] = pd.DataFrame(columns=config.MONTHLY_COLUMNS)

    if config.YEARLY_DATA_CSV.exists():
        sheets["yearly_data"] = pd.read_csv(config.YEARLY_DATA_CSV)
    else:
        logger.warning(
            "export_outputs | %s not found — yearly_data sheet will be empty",
            config.YEARLY_DATA_CSV,
        )
        sheets["yearly_data"] = pd.DataFrame(columns=config.YEARLY_COLUMNS)

    with pd.ExcelWriter(config.MASTER_XLSX, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info(
        "export_outputs | Wrote %s  (monthly: %d rows, yearly: %d rows)",
        config.MASTER_XLSX,
        len(sheets["monthly_data"]),
        len(sheets["yearly_data"]),
    )
