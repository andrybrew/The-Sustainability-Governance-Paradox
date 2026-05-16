# Econometric Validation

   Replication code and outputs for the identification battery
   in §3.6 and §4.2.3 of the manuscript.

   ## Files

   - `run_sensitivity.py` — Bai–Perron penalty sensitivity check
     (reproduces the numbers in the §3.6.1 footnote).
   - `sensitivity_check_results.txt` — human-readable output of the
     sensitivity check, including per-series break dates at
     c ∈ {0.5, 1.0, 1.5, 2.0}.

   ## Forthcoming

   The following will be added in subsequent commits:

   - `01_bai_perron_breaks.py` — reproduces Table 13
   - `02_controlled_its.py` — reproduces Table 14
   - `03_placebo_dates.py` — reproduces Table 15
   - `04_first_differenced.py` — reproduces Table 16
   - `controls_monthly_2007_2024.csv` — macroeconomic control panel
     (Brent crude prices, Caldara–Iacoviello GPR index, OECD G20 CLI)

   ## How to run

   Network metric time-series are in the parent dataset folder:
   `../The_Sustainability_Governance_Paradox_Dataset/network properties/`

   Run from the repository root:

       python econometric_validation/run_sensitivity.py

   Requires: `pandas`, `numpy`, `ruptures`.
