# Econometric Validation

Replication code for the identification battery in §3.6 and §4.2.3 of the
manuscript *"Structural Inertia and Market Friction in Global Fossil Fuel
Trade: A Longitudinal Network Analysis of Resource Path Dependencies."*

The four analysis scripts reproduce Tables 13–16. Each loads the precomputed
monthly network-metric series from the repository's
`The_Sustainability_Governance_Paradox_Dataset/network properties/` folder and
the macroeconomic control panel produced by `fetch_controls.py`.

## Contents

| File | Purpose |
|------|---------|
| `common.py` | Shared loading and transform utilities (used by all scripts). |
| `01_bai_perron_breaks.py` | Bai–Perron / PELT structural break detection → **Table 13**, plus the penalty-sensitivity check reported in §4.2.3.1. |
| `02_controlled_its.py` | Controlled interrupted time-series regression → **Table 14**. |
| `03_placebo_dates.py` | Placebo-date grid and lag-structure check → **Table 15**. |
| `04_first_differenced.py` | First-differenced specification → **Table 16**. |
| `fetch_controls.py` | Downloads the three macroeconomic control series and writes the control panel. |
| `controls_monthly_2007_2024.csv` | Control-panel **schema** (date index + empty value columns). Populated by `fetch_controls.py` — see below. |
| `run_sensitivity.py` | Standalone penalty-sensitivity script (also runnable on its own). |
| `sensitivity_check_results.txt` | Human-readable output of the sensitivity check. |
| `requirements.txt` | Python dependencies. |

## The macroeconomic control panel

`controls_monthly_2007_2024.csv` is committed as a **schema only**: it contains
the correct monthly date index (Jan 2007 – Dec 2024, 216 rows) and three empty
value columns (`brent`, `gpr`, `growth`). The actual values are **not**
redistributed here, because the three series are maintained by third parties
under their own terms of use:

- **`brent`** — Brent crude oil price, monthly average (USD/barrel).
  FRED series [`POILBREUSDM`](https://fred.stlouisfed.org/series/POILBREUSDM).
- **`gpr`** — Caldara & Iacoviello (2022) Geopolitical Risk Index, monthly.
  [matteoiacoviello.com/gpr.htm](https://www.matteoiacoviello.com/gpr.htm).
- **`growth`** — US Industrial Production Index, monthly, used as a global
  activity proxy. FRED series [`INDPRO`](https://fred.stlouisfed.org/series/INDPRO).

To reconstruct the panel, run:

```bash
python fetch_controls.py
```

`fetch_controls.py` can pull Brent and INDPRO automatically from FRED (requires
`pandas-datareader`); the GPR series is downloaded manually and its file path set
at the top of the script. See the script's header for the exact sources, column
expectations, and run instructions. The script aligns all three series to
month-start dates, restricts to the Jan 2007 – Dec 2024 window, and overwrites
`controls_monthly_2007_2024.csv` with the populated panel.

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build the control panel (see "The macroeconomic control panel" above)
python fetch_controls.py

# 3. Reproduce the tables (run from this folder)
python 01_bai_perron_breaks.py     # -> outputs/table_13_breaks.csv
python 02_controlled_its.py        # -> outputs/table_14_its.csv
python 03_placebo_dates.py         # -> outputs/table_15_placebo.csv
python 04_first_differenced.py     # -> outputs/table_16_differenced.csv
```

Each script prints its table to the console and writes a CSV to `outputs/`.

## Method summary

All four scripts implement the specification described in §3.6 of the manuscript:

- **Break detection** (§3.6.1): PELT with an L2 cost function and a BIC-style
  penalty β = c · σ̂² · log(n), c = 1, σ̂² estimated from the residuals of a
  fitted linear trend, minimum segment length 12 months. Robustness to the
  penalty constant is checked for c ∈ {0.5, 1.0, 1.5, 2.0}.
- **Transforms** (§3.6.2): density and modularity are logit-transformed,
  average weighted degree is log-transformed, node count is left in raw form.
- **Controlled ITS** (§3.6.2): OLS with Newey–West HAC standard errors at lag 6;
  the headline test is the joint F-test H₀: β₂ = β₃ = 0.
- **Robustness** (§3.6.3): placebo intervention dates, three-month-lagged
  controls, and a first-differenced specification.

## Notes on reproducibility

Structural-break placement can depend on the exact change-point library and
penalty calibration. Minor differences in detected break dates across software
versions do not affect the central finding — that no break localizes to the
SDG adoption window and that the controlled effect does not survive the placebo
and first-differenced checks — which is robust across the penalty range
documented in `sensitivity_check_results.txt`.
