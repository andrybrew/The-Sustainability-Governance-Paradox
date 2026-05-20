"""
04_first_differenced.py — reproduces Table 16.

First-differenced controlled specification, implementing the third robustness
check in manuscript section 3.6.3. The model is re-estimated in differences:

    dlog(Y_t) = a0 + a1*SDG_t + a2*(post-SDG slope)
                   + d1*dlog(Brent_t) + d2*dlog(GPR_t) + d3*dlog(Growth_t) + e_t

Differencing removes level shifts and the deterministic trend. If the apparent
post-2015 effect in the level specification (Table 14) reflects a one-time
regime change rather than an ongoing policy-driven process, it should NOT
survive differencing — i.e., the joint SDG test should be insignificant in
nearly all specifications.

For node_count (estimated in raw form in the level model) the first difference
is the simple month-over-month change; for the logit/log metrics the first
difference is taken on the transformed series.

Usage:
    python 04_first_differenced.py

Requires: numpy, pandas, statsmodels.
Inputs:   controls_monthly_2007_2024.csv  (run fetch_controls.py first)
Outputs:  outputs/table_16_differenced.csv
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

import statsmodels.api as sm

from common import (
    load_metric_series, load_controls, transform,
    SDG_DATE, COMMODITIES, METRICS,
)

HAC_LAG = 6
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def main():
    series = load_metric_series()
    controls = load_controls()

    rows = []
    for commodity in COMMODITIES:
        for metric in METRICS:
            s = series.get((commodity, metric))
            if s is None:
                continue
            ts = transform(metric, s)

            df = pd.DataFrame({"Y": ts}).join(controls, how="inner").dropna()

            # First differences of the (already transformed) outcome and of
            # the log-controls.
            dY = df["Y"].diff()
            dlog_brent = np.log(df["brent"]).diff()
            dlog_gpr = np.log(df["gpr"]).diff()
            dlog_growth = np.log(df["growth"]).diff()

            work = pd.DataFrame({
                "dY": dY,
                "dlog_brent": dlog_brent,
                "dlog_gpr": dlog_gpr,
                "dlog_growth": dlog_growth,
            }).dropna()

            n = len(work)
            t = np.arange(n)
            sdg = (work.index >= SDG_DATE).astype(float).values
            t_idx = np.searchsorted(work.index, SDG_DATE)
            post_slope = np.where(sdg == 1, t - t_idx, 0.0)

            X = pd.DataFrame({
                "SDG": sdg,
                "post_slope": post_slope,
                "dlog_brent": work["dlog_brent"].values,
                "dlog_gpr": work["dlog_gpr"].values,
                "dlog_growth": work["dlog_growth"].values,
            }, index=work.index)
            X = sm.add_constant(X)

            res = sm.OLS(work["dY"].values, X.values).fit(
                cov_type="HAC", cov_kwds={"maxlags": HAC_LAG})
            names = list(X.columns)
            coef = dict(zip(names, res.params))

            R = np.zeros((2, len(names)))
            R[0, names.index("SDG")] = 1
            R[1, names.index("post_slope")] = 1
            joint_p = float(np.ravel(res.f_test(R).pvalue))

            rows.append({
                "commodity": commodity,
                "metric": metric,
                "SDG_level": f"{coef['SDG']:+.4f}",
                "SDG_slope": f"{coef['post_slope']:+.4f}",
                "joint_p": f"{joint_p:.3f}",
                "R2": f"{res.rsquared:.2f}",
                "joint_sig_5pct": "yes" if joint_p < 0.05 else "no",
            })

    table16 = pd.DataFrame(rows)
    table16.to_csv(OUT_DIR / "table_16_differenced.csv", index=False)

    n_sig = (table16["joint_sig_5pct"] == "yes").sum()
    print("=" * 70)
    print("Table 16 — first-differenced controlled specification")
    print("=" * 70)
    print(table16.to_string(index=False))
    print()
    print(f"Joint SDG test significant at 5%: {n_sig} of {len(table16)} specifications")
    print(f"Wrote {OUT_DIR/'table_16_differenced.csv'}")


if __name__ == "__main__":
    main()
