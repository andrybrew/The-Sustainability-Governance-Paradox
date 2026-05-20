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
            (download the monthly data file 'data_gpr_export.xls'/'.csv')

3. growth — OECD G20 Composite Leading Indicator, amplitude-adjusted, monthly
            OECD.Stat -> Composite Leading Indicators (MEI)
            https://stats.oecd.org  (or the OECD Data Explorer CLI export)

Why this script rather than a committed data file
--------------------------------------------------
The three series are maintained by third parties (FRED, the GPR authors,
and the OECD) under their own terms of use; this repository does not
redistribute them. Running this script reconstructs the exact panel used in
the paper from the original public sources, which is both the reproducible
and the license-respecting way to provide the control data.

Two ways to run
---------------
A) Automatic (FRED for Brent, manual files for GPR and OECD):
   - Install pandas_datareader:  pip install pandas-datareader
   - Provide GPR_CSV and OECD_CSV paths below (downloaded manually).
   - Run:  python fetch_controls.py

B) Fully manual:
   - Download all three series as CSVs.
   - Set BRENT_CSV, GPR_CSV, OECD_CSV to their paths.
   - Run:  python fetch_controls.py

In either case the script aligns everything to month-start dates, restricts to
Jan 2007 - Dec 2024 (216 rows), forward/back-fills nothing (gaps are reported),
and writes controls_monthly_2007_2024.csv.

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
# Configure these paths if you downloaded the series manually.
# Leave BRENT_CSV = None to fetch Brent automatically from FRED.
# ----------------------------------------------------------------------
BRENT_CSV = None          # e.g. "POILBREUSDM.csv"  (columns: DATE, POILBREUSDM)
GPR_CSV = "gpr_monthly.csv"   # Caldara-Iacoviello export (columns: month, GPR)
OECD_CSV = "oecd_g20_cli.csv" # OECD CLI export (columns: TIME, Value)


def monthly_index():
    return pd.date_range(START, END, freq="MS")


def fetch_brent():
    if BRENT_CSV:
        df = pd.read_csv(BRENT_CSV)
        df.columns = [c.strip().lower() for c in df.columns]
        date_col = [c for c in df.columns if "date" in c][0]
        val_col = [c for c in df.columns if c not in (date_col,)][0]
        s = pd.Series(pd.to_numeric(df[val_col], errors="coerce").values,
                      index=pd.to_datetime(df[date_col]))
    else:
        try:
            from pandas_datareader import data as web
        except ImportError:
            sys.exit("Install pandas-datareader (pip install pandas-datareader) "
                     "or set BRENT_CSV to a manually downloaded file.")
        s = web.DataReader("POILBREUSDM", "fred", START, END)["POILBREUSDM"]
    s.index = s.index.to_period("M").to_timestamp("MS")
    return s.rename("brent")


def fetch_gpr():
    df = pd.read_csv(GPR_CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = [c for c in df.columns if c in ("month", "date", "time")][0]
    val_col = [c for c in df.columns if "gpr" in c][0]
    s = pd.Series(pd.to_numeric(df[val_col], errors="coerce").values,
                  index=pd.to_datetime(df[date_col]))
    s.index = s.index.to_period("M").to_timestamp("MS")
    return s.rename("gpr")


def fetch_oecd_cli():
    df = pd.read_csv(OECD_CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = [c for c in df.columns if c in ("time", "date", "month")][0]
    val_col = [c for c in df.columns if c in ("value", "obs_value", "cli")][0]
    s = pd.Series(pd.to_numeric(df[val_col], errors="coerce").values,
                  index=pd.to_datetime(df[date_col]))
    s.index = s.index.to_period("M").to_timestamp("MS")
    return s.rename("growth")


def main():
    idx = monthly_index()
    panel = pd.DataFrame(index=idx)
    panel.index.name = "date"

    for name, fetch in (("brent", fetch_brent),
                        ("gpr", fetch_gpr),
                        ("growth", fetch_oecd_cli)):
        try:
            s = fetch().reindex(idx)
            panel[name] = s.values
            missing = panel[name].isna().sum()
            print(f"{name:8s}: {len(s.dropna())} obs loaded, {missing} missing in range")
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


if __name__ == "__main__":
    main()
