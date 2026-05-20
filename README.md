# The Sustainability Governance Paradox

Replication data and code for the manuscript:

> **Structural Inertia and Market Friction in Global Fossil Fuel Trade: A Longitudinal Network Analysis of Resource Path Dependencies**

A longitudinal network analysis of global coal, crude oil, and natural gas trade networks (2007–2024), examining whether fossil-fuel trade architectures structurally reconfigured in response to the Sustainable Development Goals (SDGs), and distinguishing policy-driven change from change driven by exogenous macroeconomic shocks.

The study uses 216 monthly bilateral trade observations from the UN Comtrade Database to construct directed, weighted trade networks for each commodity, computes structural and centrality diagnostics, and applies a four-step econometric identification battery (Bai–Perron structural breaks, controlled interrupted time-series, placebo intervention dates, and a first-differenced specification) to test whether the January 2015 SDG adoption coincides with a structural break that is distinguishable from the contemporaneous 2014–2016 oil-price collapse.

## Repository structure

```
.
├── The_Sustainability_Governance_Paradox_Dataset/
│   ├── network properties/        # Monthly network metrics (density, modularity,
│   │                              #   avg weighted degree, node count) for all
│   │                              #   three commodities, 2007–2024
│   ├── coal/                      # Pre- and post-SDG edge lists + Gephi files
│   ├── crude_oil/                 # Pre- and post-SDG edge lists + Gephi files
│   └── natural_gas/               # Pre- and post-SDG edge lists + Gephi files
│
├── econometric_validation/        # Replication code for the identification battery
│   ├── 01_bai_perron_breaks.py    #   -> Table 13 (structural breaks)
│   ├── 02_controlled_its.py       #   -> Table 14 (controlled ITS)
│   ├── 03_placebo_dates.py        #   -> Table 15 (placebo grid)
│   ├── 04_first_differenced.py    #   -> Table 16 (first-differenced)
│   ├── fetch_controls.py          #   builds the macroeconomic control panel
│   ├── controls_monthly_2007_2024.csv  # control-panel schema (populate via fetch_controls.py)
│   ├── common.py                  #   shared loading/transform utilities
│   ├── run_sensitivity.py         #   penalty-sensitivity check
│   └── README.md                  #   detailed instructions for the replication code
│
├── LICENSE
└── README.md                      # this file
```

## Data

All trade data were originally sourced from the publicly accessible
[UN Comtrade Database](https://comtradeplus.un.org/) and cover monthly bilateral
trade records for coal (HS 2701), crude oil (HS 2709), and natural gas (HS 2711)
between January 2007 and December 2024. Records were standardized and cleaned
prior to network construction.

The macroeconomic control series used in the identification analysis (Brent crude
prices, the Caldara–Iacoviello Geopolitical Risk Index, and the OECD G20 Composite
Leading Indicator) are maintained by third parties and are not redistributed here;
`econometric_validation/fetch_controls.py` reconstructs the control panel from the
original public sources.

## Reproducing the analysis

See [`econometric_validation/README.md`](econometric_validation/README.md) for
full instructions. In brief:

```bash
cd econometric_validation
pip install -r requirements.txt
python fetch_controls.py          # build the macroeconomic control panel
python 01_bai_perron_breaks.py    # Table 13
python 02_controlled_its.py       # Table 14
python 03_placebo_dates.py        # Table 15
python 04_first_differenced.py    # Table 16
```

## Citation

If you use this data or code, please cite the manuscript. Citation details will be
added upon publication.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
