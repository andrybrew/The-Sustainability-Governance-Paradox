"""
03_placebo_dates.py — reproduces Table 15.

Placebo-date robustness check for the controlled ITS, implementing the first
robustness check described in manuscript section 3.6.3.

For each commodity-metric pair, the controlled ITS is re-estimated treating
each of seven candidate dates as the "intervention":

    January 2009, 2011, 2013, 2015 (the real SDG date), 2017, 2019, 2021

The cell value is the joint F-test p-value (H0: level = slope = 0) at that
date. If arbitrary placebo dates yield joint tests as significant as the true
January 2015 date, the SDG dummy is absorbing generic post-period drift rather
than a 2015-specific effect.

The script also runs the lag-structure check (replacing contemporaneous
controls with their three-month lags) and reports the share of significant
joint tests.

Usage:
    python 03_placebo_dates.py

Requires: numpy, pandas, statsmodels.
Inputs:   controls_monthly_2007_2024.csv  (run fetch_controls.py first)
Outputs:  outputs/table_15_placebo.csv
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

import statsmodels.api as sm

from common import (
    load_metric_series, load_controls, transform,
    COMMODITIES, METRICS,
)

HAC_LAG = 6
PLACEBO_DATES = [pd.Timestamp(f"{y}-01-01")
                 for y in (2009, 2011, 2013, 2015, 2017, 2019, 2021)]
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def joint_p_at_date(ts: pd.Series, controls: pd.DataFrame,
                    intervention: pd.Timestamp, lag_controls: bool = False):
    """Estimate the controlled ITS treating `intervention` as the break and
    return the joint F-test p-value."""
    df = pd.DataFrame({"Y": ts}).join(controls, how="inner").dropna()
    if lag_controls:
        df[["brent", "gpr", "growth"]] = df[["brent", "gpr", "growth"]].shift(3)
        df = df.dropna()

    n = len(df)
    t = np.arange(n)
    sdg = (df.index >= intervention).astype(float).values
    if sdg.sum() < 6 or (n - sdg.sum()) < 6:
        return np.nan  # not enough observations on one side
    t_idx = np.searchsorted(df.index, intervention)
    post_slope = np.where(sdg == 1, t - t_idx, 0.0)

    X = pd.DataFrame({
        "t": t, "SDG": sdg, "post_slope": post_slope,
        "log_brent": np.log(df["brent"].values),
        "log_gpr": np.log(df["gpr"].values),
        "log_growth": np.log(df["growth"].values),
    }, index=df.index)
    X = sm.add_constant(X)

    res = sm.OLS(df["Y"].values, X.values).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAG})
    names = list(X.columns)
    R = np.zeros((2, len(names)))
    R[0, names.index("SDG")] = 1
    R[1, names.index("post_slope")] = 1
    return float(np.ravel(res.f_test(R).pvalue))


def main():
    series = load_metric_series()
    controls = load_controls()

    # ---- Placebo grid (Table 15) ----
    rows = []
    lag_sig = 0
    lag_total = 0
    for commodity in COMMODITIES:
        for metric in METRICS:
            s = series.get((commodity, metric))
            if s is None:
                continue
            ts = transform(metric, s)
            row = {"commodity": commodity, "metric": metric}
            for d in PLACEBO_DATES:
                p = joint_p_at_date(ts, controls, d)
                row[d.strftime("%Y-%m")] = round(p, 3) if not np.isnan(p) else np.nan
            rows.append(row)

            # lag-structure check at the real SDG date
            lp = joint_p_at_date(ts, controls, pd.Timestamp("2015-01-01"),
                                 lag_controls=True)
            lag_total += 1
            if not np.isnan(lp) and lp < 0.05:
                lag_sig += 1

    table15 = pd.DataFrame(rows)
    table15.to_csv(OUT_DIR / "table_15_placebo.csv", index=False)

    # Significance counts per date
    date_cols = [d.strftime("%Y-%m") for d in PLACEBO_DATES]
    sig_counts = {c: int((table15[c] < 0.05).sum()) for c in date_cols}

    print("=" * 78)
    print("Table 15 — placebo test of the controlled ITS joint SDG test")
    print("=" * 78)
    print(table15.to_string(index=False))
    print()
    print("Significant at 5% (per date):")
    for c in date_cols:
        print(f"  {c}: {sig_counts[c]}/{len(table15)}")
    print()
    print(f"Lag-structure check (3-month lagged controls): "
          f"{lag_sig}/{lag_total} joint tests significant at 5%")
    print(f"Wrote {OUT_DIR/'table_15_placebo.csv'}")


if __name__ == "__main__":
    main()
