"""
common.py — shared utilities for the econometric validation scripts.

Loads the precomputed monthly network-metric series from the repository's
`network properties` folder, applies the transformations described in
manuscript section 3.6, and exposes helpers used by all four analysis
scripts (01–04).

The four metric files are wide-format CSVs (one row per commodity, one
column per month, semicolon-separated) located at:

    ../The_Sustainability_Governance_Paradox_Dataset/network properties/
        density.csv
        modularity.csv
        avg weighted degree.csv
        nodes.csv

Each script imports from here so the loading/transform logic is defined once.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

# Resolve the dataset folder relative to this file's location in the repo.
REPO_ROOT = Path(__file__).resolve().parent.parent
NETWORK_DIR = REPO_ROOT / "The_Sustainability_Governance_Paradox_Dataset" / "network properties"
CONTROLS_CSV = Path(__file__).resolve().parent / "controls_monthly_2007_2024.csv"

METRIC_FILES = {
    "density": "density.csv",
    "modularity": "modularity.csv",
    "avg_weighted_degree": "avg weighted degree.csv",
    "node_count": "nodes.csv",
}

# Row labels as they appear in the wide-format CSVs -> canonical commodity key
COMMODITY_LABELS = {
    "coal": "coal",
    "crude oil": "crude_oil",
    "crudeoil": "crude_oil",
    "crude_oil": "crude_oil",
    "natural gas": "natural_gas",
    "gas": "natural_gas",
    "natural_gas": "natural_gas",
}

COMMODITIES = ["coal", "crude_oil", "natural_gas"]
METRICS = ["density", "modularity", "avg_weighted_degree", "node_count"]

SDG_DATE = pd.Timestamp("2015-01-01")


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def _parse_month_columns(cols) -> pd.DatetimeIndex:
    """Parse 'Jan-07'-style month headers into a DatetimeIndex."""
    return pd.to_datetime(list(cols), format="%b-%y")


def load_metric_series() -> dict:
    """
    Return {(commodity, metric): pd.Series} with a monthly DatetimeIndex,
    spanning Jan 2007 – Dec 2024 (216 observations per series).
    """
    series = {}
    for metric, filename in METRIC_FILES.items():
        path = NETWORK_DIR / filename
        df = pd.read_csv(path, sep=";", encoding="utf-8")
        label_col = df.columns[0]
        date_cols = df.columns[1:]
        dates = _parse_month_columns(date_cols)
        for _, row in df.iterrows():
            raw = row[label_col]
            key = COMMODITY_LABELS.get(str(raw).strip().lower())
            if key is None:
                continue
            values = pd.to_numeric(row[date_cols].values, errors="coerce")
            series[(key, metric)] = pd.Series(values, index=dates,
                                              name=f"{key}_{metric}")
    return series


def load_controls() -> pd.DataFrame:
    """
    Load the macroeconomic control panel produced by fetch_controls.py.
    Expects columns: date, brent, gpr, growth (216 monthly rows).
    """
    if not CONTROLS_CSV.exists():
        raise FileNotFoundError(
            f"{CONTROLS_CSV.name} not found. Run fetch_controls.py first to "
            "populate the macroeconomic control panel."
        )
    df = pd.read_csv(CONTROLS_CSV)
    if df["brent"].isna().all():
        raise ValueError(
            f"{CONTROLS_CSV.name} contains only the schema (empty value "
            "columns). Run fetch_controls.py to populate it before running "
            "the analysis scripts."
        )
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


# ----------------------------------------------------------------------
# Transforms (manuscript section 3.6.1 / 3.6.2)
# ----------------------------------------------------------------------

def logit(x: np.ndarray) -> np.ndarray:
    """Logit transform with clamping to keep values strictly in (0,1)."""
    x = np.clip(np.asarray(x, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def transform(metric: str, s: pd.Series) -> pd.Series:
    """
    Apply the manuscript transformation:
      density, modularity  -> logit
      avg_weighted_degree  -> log(x + 1)
      node_count           -> identity
    """
    s = s.astype(float)
    if metric in ("density", "modularity"):
        return pd.Series(logit(s.values), index=s.index, name=s.name)
    if metric == "avg_weighted_degree":
        return pd.Series(np.log(s.values + 1.0), index=s.index, name=s.name)
    if metric == "node_count":
        return s
    raise ValueError(f"Unknown metric: {metric}")


# ----------------------------------------------------------------------
# Crisis windows (manuscript section 3.6.1)
# ----------------------------------------------------------------------

CRISIS_WINDOWS = {
    "GFC 2008-09": (pd.Timestamp("2008-09-01"), pd.Timestamp("2009-12-31")),
    "Oil collapse 2014-16": (pd.Timestamp("2014-06-01"), pd.Timestamp("2016-06-30")),
    "COVID-19": (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-08-31")),
    "Russia-Ukraine": (pd.Timestamp("2022-02-01"), pd.Timestamp("2022-09-30")),
}

SDG_WINDOW = (SDG_DATE - pd.DateOffset(months=6),
              SDG_DATE + pd.DateOffset(months=6))


def crisis_match(date: pd.Timestamp):
    """Return the crisis-window name a break date falls in, or None."""
    for name, (lo, hi) in CRISIS_WINDOWS.items():
        if lo <= date <= hi:
            return name
    return None


def in_sdg_window(date: pd.Timestamp) -> bool:
    return SDG_WINDOW[0] <= date <= SDG_WINDOW[1]
