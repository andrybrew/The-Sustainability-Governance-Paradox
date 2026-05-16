"""
Bai-Perron / PELT penalty sensitivity check — actual data run.

Loads from the four wide-format CSVs in /home/claude/sensitivity/data/
which were extracted directly from the GitHub repository.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import ruptures as rpt

# ============================================================
# CONFIGURATION (matches V2 sec 3.6.1)
# ============================================================

SDG_DATE = pd.Timestamp("2015-01-01")
SDG_WINDOW_MONTHS = 6
MIN_SEGMENT_LENGTH = 12
PENALTY_VALUES = [0.5, 1.0, 1.5, 2.0]

DATA_DIR = Path(__file__).parent.parent / "The_Sustainability_Governance_Paradox_Dataset" / "network properties"
FILES = {
    "density": "density.csv",
    "modularity": "modularity.csv",
    "avg_weighted_degree": "avg_weighted_degree.csv",
    "node_count": "nodes.csv",
}
COMMODITY_LABELS = {
    "Coal": "coal", "Coal ": "coal",
    "Crude Oil": "crude_oil",
    "Natural Gas": "natural_gas",
}

# ============================================================
# LOAD - wide format CSVs (rows=commodity, cols=monthly date)
# ============================================================

def load_series():
    series = {}
    for metric, filename in FILES.items():
        df = pd.read_csv(DATA_DIR / filename, sep=";", encoding="utf-8")
        # First column is "period" with the row labels (Coal, Crude Oil, Natural Gas)
        period_col = df.columns[0]
        date_cols = df.columns[1:]
        # Parse the date column headers as monthly periods
        dates = pd.to_datetime(date_cols, format="%b-%y")
        for _, row in df.iterrows():
            label = row[period_col].strip() if isinstance(row[period_col], str) else row[period_col]
            commodity = COMMODITY_LABELS.get(label)
            if commodity is None:
                print(f"  Skipping unrecognised row label: {label!r}")
                continue
            values = pd.to_numeric(row[date_cols].values, errors="coerce")
            s = pd.Series(values, index=dates, name=f"{commodity}_{metric}")
            series[(commodity, metric)] = s
    return series

# ============================================================
# TRANSFORM (V2 sec 3.6.1)
# ============================================================

def logit(x):
    x_safe = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x_safe / (1 - x_safe))

def transform_series(metric, series):
    s = series.astype(float)
    if metric in ("density", "modularity"):
        return pd.Series(logit(s.values), index=s.index)
    elif metric == "avg_weighted_degree":
        return pd.Series(np.log(s.values + 1.0), index=s.index)
    elif metric == "node_count":
        return s
    raise ValueError(metric)

# ============================================================
# BAI-PERRON / PELT
# ============================================================

def detect_breaks(transformed, c):
    y = transformed.values.astype(float)
    # Handle any NaNs from coercion
    if np.any(np.isnan(y)):
        mask = ~np.isnan(y)
        y_clean = y[mask]
        index_clean = transformed.index[mask]
    else:
        y_clean = y
        index_clean = transformed.index
    n = len(y_clean)
    t = np.arange(n)
    slope, intercept = np.polyfit(t, y_clean, 1)
    residuals = y_clean - (slope * t + intercept)
    sigma_hat_sq = np.var(residuals, ddof=2)
    beta = c * sigma_hat_sq * np.log(n)
    algo = rpt.Pelt(model="l2", min_size=MIN_SEGMENT_LENGTH).fit(y_clean)
    bps = algo.predict(pen=beta)
    if bps and bps[-1] == n:
        bps = bps[:-1]
    return bps, beta, index_clean

def bp_dates(bps, index_clean):
    return [index_clean[bp] for bp in bps if bp < len(index_clean)]

def in_sdg_window(dates):
    lo = SDG_DATE - pd.DateOffset(months=SDG_WINDOW_MONTHS)
    hi = SDG_DATE + pd.DateOffset(months=SDG_WINDOW_MONTHS)
    return any(lo <= d <= hi for d in dates)

# ============================================================
# MAIN
# ============================================================

print("Loading series from GitHub-extracted CSVs...")
series_dict = load_series()
print(f"Loaded {len(series_dict)} series:")
for k, s in series_dict.items():
    print(f"  {k}: n={len(s)} from {s.index[0].strftime('%Y-%m')} to {s.index[-1].strftime('%Y-%m')}")

print("\nApplying V2 sec 3.6.1 transformations...")
transformed = {k: transform_series(metric, s) for (commodity, metric), s in series_dict.items()
               for k in [(commodity, metric)]}

print("\nRunning sensitivity grid over c in", PENALTY_VALUES)
print("=" * 90)

per_c_summary = {}
per_c_details = {}

for c in PENALTY_VALUES:
    total = 0
    sdg_window_breaks = 0
    details = []
    for (commodity, metric), s in transformed.items():
        bps, beta, idx = detect_breaks(s, c)
        dates = bp_dates(bps, idx)
        total += len(dates)
        if in_sdg_window(dates):
            sdg_window_breaks += 1
        details.append({
            "series": f"{commodity}__{metric}",
            "n_breaks": len(dates),
            "dates": [d.strftime("%Y-%m") for d in dates],
            "beta": beta,
        })
    per_c_summary[c] = {"total": total, "sdg_window": sdg_window_breaks}
    per_c_details[c] = details

# === Summary table ===
print(f"\n{'c':<8}{'total breaks':<18}{'series w/ SDG-window break':<32}")
print("-" * 90)
for c in PENALTY_VALUES:
    s = per_c_summary[c]
    print(f"{c:<8}{s['total']:<18}{s['sdg_window']} / 12")

# === Per-series, per-c table ===
print("\nPer-series breakdown:")
print(f"{'series':<35}" + "".join(f"  c={c:<5}" for c in PENALTY_VALUES))
print("-" * 90)
series_names = [d["series"] for d in per_c_details[1.0]]
for sn in series_names:
    line = f"{sn:<35}"
    for c in PENALTY_VALUES:
        d = next(x for x in per_c_details[c] if x["series"] == sn)
        line += f"  {d['n_breaks']:<5}"
    print(line)

# === Detailed break dates at c=1.0 (baseline) ===
print("\nBreak dates at baseline (c=1.0):")
print("-" * 90)
for d in per_c_details[1.0]:
    if d["dates"]:
        print(f"  {d['series']:<35}: {', '.join(d['dates'])}")
    else:
        print(f"  {d['series']:<35}: (no breaks)")

# === The numbers for the manuscript ===
print()
print("=" * 90)
print("NUMBERS FOR THE V2 FOOTNOTE (item 2.1)")
print("=" * 90)
baseline_total = per_c_summary[1.0]["total"]
totals = [per_c_summary[c]["total"] for c in PENALTY_VALUES]
deltas_from_baseline = [per_c_summary[c]["total"] - baseline_total
                        for c in PENALTY_VALUES if c != 1.0]
print(f"Baseline (c=1.0):       {baseline_total} total breaks across 12 series")
print(f"Range of totals:        [{min(totals)}, {max(totals)}]")
print(f"Delta from baseline:    {[(c, per_c_summary[c]['total'] - baseline_total) for c in PENALTY_VALUES if c != 1.0]}")
print(f"Magnitude in either dir: at most {max(abs(d) for d in deltas_from_baseline)}")

print("\nSDG-window stability across penalty values:")
sdg_counts = [per_c_summary[c]["sdg_window"] for c in PENALTY_VALUES]
if all(x == 0 for x in sdg_counts):
    print("  CONFIRMED: zero series have a break in +/- 6 months of Jan 2015,")
    print("             across all four penalty values.")
elif all(x == sdg_counts[0] for x in sdg_counts):
    print(f"  STABLE: {sdg_counts[0]} of 12 series have a SDG-window break at every c.")
else:
    print(f"  VARIES: SDG-window break counts are {sdg_counts} for c={PENALTY_VALUES}")
    print("          - investigate before finalising footnote.")
