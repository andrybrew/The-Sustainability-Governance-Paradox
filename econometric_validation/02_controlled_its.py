"""
02_controlled_its.py — reproduces Table 14.

Controlled interrupted time-series (ITS) regression for each commodity-metric
pair, implementing the specification in manuscript section 3.6.2:

    Y_t = b0 + b1*t + b2*SDG_t + b3*(t - t_SDG)*SDG_t
              + g1*log(Brent_t) + g2*log(GPR_t) + g3*log(Growth_t) + e_t

where SDG_t = 1 from January 2015 onward. Estimation is OLS with Newey-West
HAC standard errors at lag 6. The reported headline test is the joint F-test
of H0: b2 = b3 = 0.

Transforms (section 3.6.2):
    density, modularity  -> logit
    avg_weighted_degree  -> log
    node_count           -> raw

Usage:
    python 02_controlled_its.py

Requires: numpy, pandas, statsmodels.
Inputs:   controls_monthly_2007_2024.csv  (run fetch_controls.py first)
Outputs:  outputs/table_14_its.csv
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


def build_design(ts: pd.Series, controls: pd.DataFrame) -> pd.DataFrame:
    """Assemble the regression design matrix for one series."""
    df = pd.DataFrame({"Y": ts})
    df = df.join(controls, how="inner")  # align on monthly date index
    df = df.dropna()

    n = len(df)
    t = np.arange(n)
    sdg = (df.index >= SDG_DATE).astype(float).values
    t_sdg_index = np.searchsorted(df.index, SDG_DATE)
    post_slope = np.where(sdg == 1, t - t_sdg_index, 0.0)

    X = pd.DataFrame({
        "t": t,
        "SDG": sdg,
        "post_slope": post_slope,
        "log_brent": np.log(df["brent"].values),
        "log_gpr": np.log(df["gpr"].values),
        "log_growth": np.log(df["growth"].values),
    }, index=df.index)
    X = sm.add_constant(X)
    return df["Y"], X


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


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
            y, X = build_design(ts, controls)

            model = sm.OLS(y.values, X.values)
            res = model.fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAG})

            names = list(X.columns)
            coef = dict(zip(names, res.params))
            pval = dict(zip(names, res.pvalues))

            # Joint F-test of H0: SDG = post_slope = 0
            R = np.zeros((2, len(names)))
            R[0, names.index("SDG")] = 1
            R[1, names.index("post_slope")] = 1
            joint = res.f_test(R)
            joint_p = float(np.ravel(joint.pvalue))

            rows.append({
                "commodity": commodity,
                "metric": metric,
                "SDG_lvl": f"{coef['SDG']:+.3f}{stars(pval['SDG'])}",
                "SDG_slp": f"{coef['post_slope']:+.4f}{stars(pval['post_slope'])}",
                "Brent": f"{coef['log_brent']:+.3f}{stars(pval['log_brent'])}",
                "GPR": f"{coef['log_gpr']:+.3f}{stars(pval['log_gpr'])}",
                "Growth": f"{coef['log_growth']:+.3f}{stars(pval['log_growth'])}",
                "R2": f"{res.rsquared:.2f}",
                "joint_p": f"{joint_p:.3f}",
                "joint_sig_5pct": "yes" if joint_p < 0.05 else "no",
            })

    table14 = pd.DataFrame(rows)
    table14.to_csv(OUT_DIR / "table_14_its.csv", index=False)

    n_sig = (table14["joint_sig_5pct"] == "yes").sum()
    print("=" * 70)
    print("Table 14 — controlled ITS regression results")
    print("=" * 70)
    print(table14.to_string(index=False))
    print()
    print(f"Joint SDG test significant at 5%: {n_sig} of {len(table14)} specifications")
    print(f"Wrote {OUT_DIR/'table_14_its.csv'}")


if __name__ == "__main__":
    main()
