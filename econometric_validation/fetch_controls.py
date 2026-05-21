"""
fetch_controls.py — build the macroeconomic control panel.

Downloads the three control series used in the controlled ITS (section 3.6.2)
and writes them, aligned to a monthly Jan 2007 - Dec 2024 index, to:

    controls_monthly_2007_2024.csv   (columns: date, brent, gpr, growth)

Sources
-------
1. brent  — Brent crude oil price, monthly average (USD/barrel)
            FRED series POILBREUSDM
            https://fred.stlouisfed.org/series/POILBREUSDM

2. gpr    — Caldara & Iacoviello (2022) Geopolitical Risk Index, monthly (GPR)
            https://www.matteoiacoviello.com/gpr.htm
            (download the monthly data file 'data_gpr_export.xls')

3. growth — US Industrial Production Index, monthly, used as a global activity
            proxy. FRED series INDPRO
            https://fred.stlouisfed.org/series/INDPRO

Why this script rather than a committed data file
--------------------------------------------------
The three series are maintained by third parties (FRED and the GPR authors)
under their own terms of use; this repository does not redistribute the raw
source files. Running this script reconstructs the exact panel used in the
paper from the original public sources, which is both the reproducible and the
license-respecting way to provide the control data. (A populated
controls_monthly_2007_2024.csv is also committed for convenience.)

Two ways to run
---------------
A) Automatic (Brent and INDPRO from FRED, GPR from a manually downloaded file):
   - Install pandas_datareader:  pip install pandas-datareader
   - Download the GPR monthly file and set GPR_FILE below.
   - Run:  python fetch_controls.py

B) Fully manual:
   - Download all three series.
   - Set BRENT_CSV, GPR_FILE, INDPRO_CSV to their paths.
   - Run:  python fetch_controls.py

In either case the script aligns everything to month-start dates, restricts to
Jan 2007 - Dec 2024 (216 rows), reports any gaps, and writes
controls_monthly_2007_2024.csv.

After running, verify the panel by re-running 02_controlled_its.py and
confirming the output matches Table 14 in the manuscript.
"""

from __future__ import annotations
import sys
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent / "controls_monthly_2007_2024.csv"
START = pd.Timestamp("2007-01-01")
END = pd.Timestamp("2024-12-01")

# ----------------------------------------------------------------------
# Configure these paths if you downloaded a series manually.
# Leave BRENT_CSV / INDPRO_CSV = None to fetch them automatically from FRED.
# GPR must be downloaded manually from matteoiacoviello.com/gpr.htm.
# ----------------------------------------------------------------------
BRENT_CSV = None                    # e.g. "POILBREUSDM.csv"
INDPRO_CSV = None                   # e.g. "INDPRO.csv"
GPR_FILE = "data_gpr_export.xls"    # Caldara-Iacoviello monthly export


def monthly_index():
    return pd.date_range(START, END, freq="MS")


def _fred_csv(path, series_id):
    """Parse a FRED CSV export (columns: observation_date, <SERIES_ID>)."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    date_col = [c for c in df.columns if "date" in c.lower()][0]
    val_col = [c for c in df.columns if c != date_col][0]
    s = pd.Series(pd.to_numeric(df[val_col], errors="coerce").values,
                  index=pd.to_datetime(df[date_col]))
    s.index = s.index.to_period("M").to_timestamp()
    return s


def fetch_fred(series_id, manual_path):
    if manual_path:
        return _fred_csv(manual_path, series_id)
    try:
        from pandas_datareader import data as web
    except ImportError:
        sys.exit(f"Install pandas-datareader (pip install pandas-datareader) "
                 f"or set the manual CSV path for {series_id}.")
    s = web.DataReader(series_id, "fred", START, END)[series_id]
    s.index = s.index.to_period("M").to_timestamp()
    return s


def fetch_brent():
    return fetch_fred("POILBREUSDM", BRENT_CSV).rename("brent")


def fetch_growth():
    return fetch_fred("INDPRO", INDPRO_CSV).rename("growth")


def fetch_gpr():
    """Parse the Caldara-Iacoviello monthly Excel export (columns: month, GPR)."""
    df = pd.read_excel(GPR_FILE)
    df.columns = [str(c).strip() for c in df.columns]
    date_col = [c for c in df.columns if c.lower() in ("month", "date", "time")][0]
    val_col = "GPR" if "GPR" in df.columns else \
        [c for c in df.columns if c.upper() == "GPR"][0]
    s = pd.Series(pd.to_numeric(df[val_col], errors="coerce").values,
                  index=pd.to_datetime(df[date_col]))
    s.index = s.index.to_period("M").to_timestamp()
    return s.rename("gpr")


def main():
    idx = monthly_index()
    panel = pd.DataFrame(index=idx)
    panel.index.name = "date"

    for name, fetch in (("brent", fetch_brent),
                        ("gpr", fetch_gpr),
                        ("growth", fetch_growth)):
        try:
            s = fetch().reindex(idx)
            panel[name] = s.values
            missing = panel[name].isna().sum()
            print(f"{name:8s}: loaded, {missing} missing in 2007-2024 window")
        except FileNotFoundError as e:
            print(f"{name:8s}: source file not found ({e}). "
                  f"Column left empty — see header for download instructions.")
            panel[name] = pd.NA

    panel.reset_index().to_csv(OUT, index=False)
    print(f"\nWrote {OUT} ({len(panel)} rows).")
    if panel.isna().any().any():
        print("WARNING: some columns are empty or have gaps. Download the "
              "missing sources (see this script's header) and re-run before "
              "running the analysis scripts.")
    else:
        print("All three series complete with no gaps. Ready for analysis.")


if __name__ == "__main__":
    main()
