# variable-lag-tabu

Tabu-based causal structure learning for multivariate time series with edge-specific variable lags.

This repository contains the reference implementation for the paper:

> **Time series causal discovery with variable lags**  
> Bruno Petrungaro and Anthony C. Constantinou  
> arXiv:2605.04081  
> https://arxiv.org/abs/2605.04081

## Overview

Many time-series causal discovery methods require a fixed lag window or do not explicitly optimise the lag assigned to each causal edge. `variable-lag-tabu` learns a compact lagged causal graph in which each directed edge can have its own lag:

```text
X(t - lag)  --->  Y(t)

# Repository structure

variable-lag-tabu/
├── src/
│   ├── variable_lag_tabu.py            # Main sequential implementation
│   ├── variable_lag_tabu_parallel.py   # Parallel/decomposable implementation
│   ├── sweep_variable_lag_bn.py        # Synthetic simulation sweeps
│   ├── real_world_application.py       # Real-world UK COVID-19 application
│   └── analysis.py                     # Analysis helper script
├── create_venv.sh
├── pyproject.toml
├── setup.cfg
├── LICENSE
└── README.md



