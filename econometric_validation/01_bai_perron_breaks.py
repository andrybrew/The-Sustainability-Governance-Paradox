"""
01_bai_perron_breaks.py — reproduces Table 13.

Endogenous structural break detection on the twelve commodity-metric series,
implementing the procedure described in manuscript section 3.6.1:

  - PELT algorithm (Killick et al., 2012) with an L2 cost function
  - BIC-style penalty  beta = c * sigma_hat^2 * log(n),  c = 1
  - sigma_hat^2 = residual variance of a fitted linear trend
  - minimum segment length = 12 months

For each detected break the script records the date, whether it falls in one
of the four pre-specified macroeconomic crisis windows, and whether it falls
within +/- 6 months of the SDG adoption date (January 2015).

It also runs the penalty-sensitivity check reported in section 4.2.3.1
(c in {0.5, 1.0, 1.5, 2.0}).

Usage:
    python 01_bai_perron_breaks.py

Requires: numpy, pandas, ruptures.  (pip install ruptures)
Outputs:  outputs/table_13_breaks.csv
          outputs/penalty_sensitivity.csv
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

import ruptures as rpt

from common import (
    load_metric_series, transform, crisis_match, in_sdg_window,
    COMMODITIES, METRICS,
)

MIN_SEGMENT = 12
PENALTY_GRID = [0.5, 1.0, 1.5, 2.0]
BASELINE_C = 1.0

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def detect_breaks(y: np.ndarray, c: float):
    """
    Detect breakpoints in a 1-D series using PELT/L2 with the manuscript's
    BIC-style penalty. Returns a list of integer break indices (segment
    boundaries, excluding the series end).
    """
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(y)
    y = y[mask]
    n = len(y)

    # sigma_hat^2 from the residuals of a linear trend fit
    t = np.arange(n)
    slope, intercept = np.polyfit(t, y, 1)
    resid = y - (slope * t + intercept)
    sigma2 = np.var(resid, ddof=2)

    penalty = c * sigma2 * np.log(n)

    algo = rpt.Pelt(model="l2", min_size=MIN_SEGMENT).fit(y)
    bkps = algo.predict(pen=penalty)
    if bkps and bkps[-1] == n:
        bkps = bkps[:-1]
    return bkps, mask


def break_dates(bkps, index, mask):
    valid_index = index[mask]
    return [valid_index[b] for b in bkps if b < len(valid_index)]


def main():
    series = load_metric_series()

    # ---- Baseline (c = 1): build Table 13 ----
    rows = []
    for commodity in COMMODITIES:
        for metric in METRICS:
            s = series.get((commodity, metric))
            if s is None:
                continue
            ts = transform(metric, s)
            bkps, mask = detect_breaks(ts.values, BASELINE_C)
            for d in break_dates(bkps, ts.index, mask):
                cm = crisis_match(d)
                rows.append({
                    "commodity": commodity,
                    "metric": metric,
                    "break_date": d.strftime("%Y-%m"),
                    "crisis_match": cm if cm else "",
                    "sdg_aligned": "yes" if in_sdg_window(d) else "no",
                })

    table13 = pd.DataFrame(rows)
    table13.to_csv(OUT_DIR / "table_13_breaks.csv", index=False)

    total = len(table13)
    crisis_n = (table13["crisis_match"] != "").sum()
    sdg_n = (table13["sdg_aligned"] == "yes").sum()

    print("=" * 70)
    print("Table 13 — endogenously detected structural breaks (c = 1.0)")
    print("=" * 70)
    print(table13.to_string(index=False))
    print()
    print(f"Total breaks:                 {total}")
    print(f"Crisis-window matches:        {crisis_n}  ({100*crisis_n/total:.0f}%)")
    print(f"SDG-window aligned:           {sdg_n}")
    print()

    # ---- Penalty sensitivity (section 4.2.3.1) ----
    sens_rows = []
    for c in PENALTY_GRID:
        c_total = 0
        c_sdg = 0
        for commodity in COMMODITIES:
            for metric in METRICS:
                s = series.get((commodity, metric))
                if s is None:
                    continue
                ts = transform(metric, s)
                bkps, mask = detect_breaks(ts.values, c)
                dates = break_dates(bkps, ts.index, mask)
                c_total += len(dates)
                c_sdg += sum(1 for d in dates if in_sdg_window(d))
        sens_rows.append({"c": c, "total_breaks": c_total,
                          "sdg_window_breaks": c_sdg})

    sens = pd.DataFrame(sens_rows)
    sens.to_csv(OUT_DIR / "penalty_sensitivity.csv", index=False)

    print("Penalty sensitivity (section 4.2.3.1)")
    print("-" * 70)
    print(sens.to_string(index=False))
    print()
    print(f"Wrote {OUT_DIR/'table_13_breaks.csv'}")
    print(f"Wrote {OUT_DIR/'penalty_sensitivity.csv'}")


if __name__ == "__main__":
    main()
