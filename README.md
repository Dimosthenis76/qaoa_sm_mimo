# QAOA SM–MIMO Project

This repository contains all the scripts and data needed to reproduce the experiments presented in our paper on fixed‑angle QAOA for Spatial Modulation MIMO detection. Follow the instructions below to set up the environment, run the simulations, and generate all figures and CSV data files.

## Repository Structure

```
qaoa_sm_mimo/            # Root project folder
├── baseline/            # Baseline QAOA p=1 scripts
│   └── cirq_sm_mimo_baseline.py
├── grid_search/         # Grid-search tuning scripts
│   └── cirq_sm_mimo_grid.py
└── extensions/          # Data analysis and metrics scripts
    ├── effective_rate.py
    ├── mi_estimate.py
    └── sumrate_noma.py
```

## Prerequisites

* Python 3.10
* Git (to clone the repository)

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/Dimosthenis76/qaoa_sm_mimo.git
   cd qaoa_sm_mimo
   ```
2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv venv
   # macOS / Linux:
   source venv/bin/activate
   # Windows:
   venv\Scripts\activate.bat
   ```
3. **Install required packages**

   ```bash
   pip install --upgrade pip
   pip install cirq numpy scipy matplotlib
   ```

## Running the Experiments

Execute the following scripts in order, each will generate PNG figures and CSV data files in its respective folder.

1. **Baseline QAOA (p=1, fixed angles)**

   ```bash
   python baseline/cirq_sm_mimo_baseline.py
   ```

   * Outputs: `baseline/ber_vs_snr_baseline.png` (Figure)

2. **Grid‑Search Tuning**

   ```bash
   python grid_search/cirq_sm_mimo_grid.py
   ```

   * Outputs: `grid_search/ber_vs_snr_tuned.png` and `grid_search/ber_vs_snr_tuned.csv`

3. **Effective Rate Calculation**

   ```bash
   python extensions/effective_rate.py
   ```

   * Outputs: `extensions/effective_rate.png` and `extensions/effective_rate.csv`

4. **Mutual Information Estimation**

   ```bash
   python extensions/mi_estimate.py
   ```

   * Outputs: `extensions/mi_vs_snr.png` and `extensions/mi_vs_snr.csv`

5. **NOMA Sum‑Rate Analysis**

   ```bash
   python extensions/sumrate_noma.py
   ```

   * Outputs: `extensions/sumrate_noma.png` and `extensions/sumrate_noma.csv`

## Notes

* Ensure the virtual environment is active before running any script.
* All output files are overwritten on each run; back up any results if needed.
* For faster tuning, you may reduce the number of shots (`shots_tune`) in `cirq_sm_mimo_grid.py`, or number of Monte Carlo samples in `mi_estimate.py`.

## Contact

For questions or issues, please open an issue on the GitHub repository or contact the authors.
