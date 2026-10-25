# SPX Directional Signal Pipeline

[![tests](https://github.com/HD013/spx-directional-signal-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/HD013/spx-directional-signal-pipeline/actions/workflows/tests.yml)

A modular, testable Python pipeline demonstrating leakage-aware temporal validation, walk-forward evaluation, feature engineering, calibration and regime-dependent model assessment on 14 years of SPX options data (2010–2023).

The financial dataset serves as a time-dependent prediction testbed. The pipeline architecture and evaluation methods apply to other longitudinal and sequential data, such as sensor streams, monitoring data and epidemiological series.

## What this demonstrates

- **Temporal validation without leakage.** Expanding-window walk-forward evaluation with a purge at the label horizon. Features use only data available at prediction time.
- **Honest benchmarking.** Every result is reported next to an always-long baseline evaluated on the same days.
- **Calibration and regime analysis.** Reliability of predicted probabilities, and performance by volatility regime with sample sizes.
- **Tested evaluation code.** A synthetic-data test suite checks fold boundaries, feature causality and determinism, and confirms that the pipeline recovers a planted signal and finds nothing when no signal exists.
- **Modular, importable code.** Ingest, features, model and backtest are plain functions with no notebook-only logic.

## Evaluation design

| | |
|---|---|
| Training window | Expanding: all trading days before the test window, minimum 504 (~2 years) |
| Purge | 1 trading day (the label horizon) between training and test |
| Test window / step | 63 trading days (~1 quarter) / 63 days |
| Folds | 47 |
| Out-of-sample days | 2,961 (2012–2023) |
| Model | LightGBM classifier, refit on every fold |

- Each fold's model is fit on past data only. Nothing from a test window is used in training.
- Features for day *t* use only data at or before *t*: rolling features are trailing windows. The target is whether day *t+1*'s close is above day *t*'s.
- There is no fitted preprocessing (no scaling or imputation), so refitting the model per fold keeps all fitting fold-local.
- Hyperparameters and the VIX regime cutoffs (15% and 25%) were fixed before any out-of-sample evaluation and were not tuned afterwards.

## Results

Out of sample, 2,961 trading days, 2012–2023, no transaction costs.

| Metric | Model | Always long |
|---|---|---|
| Win rate | 51.3% | 53.7% |
| Sharpe (annualised) | 0.80 | 0.71 |
| Summed daily return | 159.6% | 141.2% |
| Max drawdown (summed) | -42.9% | -38.3% |

By VIX regime (VIX proxy = mean ATM call IV at 25–35 DTE):

| Regime | Days | Win rate (model) | Win rate (always long) | Sharpe (model) | Sharpe (always long) |
|---|---|---|---|---|---|
| Low (< 15%) | 1,820 | 50.3% | 53.6% | 0.15 | 0.57 |
| Medium (15–25%) | 915 | 51.9% | 53.7% | 1.04 | 0.69 |
| High (> 25%) | 173 | 55.5% | 53.8% | 2.28 | 1.66 |

**Reading the results**

- The model's win rate is below the always-long baseline, while its Sharpe is modestly higher.
- Relative to the baseline, the model underperforms when volatility is low and outperforms when it is medium or high. The high-VIX figures rest on 173 days, 154 of them in 2020 and 2022, so they reflect two market episodes and are descriptive, not conclusive.
- Predicted probabilities are poorly calibrated. The observed up-rate stays between 49% and 57% across every probability bin, and the Brier score (0.308) is worse than a constant forecast (0.249). Only the direction of each prediction is used.
- The signal is weak and sensitive to small changes: adding the 1-day purge (one fewer training row per fold) changed 13% of daily predictions.
- 53 out-of-sample days, all in 2012, have no VIX proxy because no liquid ATM option fell in the 25–35 DTE window (expiries were sparser then). They count in the overall figures but not in the regime table, which is why the regime days sum to 2,908.

## Methodology and assumptions

- **Positions:** long SPX (+1) when the predicted up-probability is at least 0.5, short (−1) otherwise. The threshold is fixed, not learned. A signal formed at day *t*'s close earns the close-to-close return from *t* to *t+1*.
- **Sharpe:** mean daily return divided by its standard deviation, times √252. No risk-free rate.
- **Returns:** "summed" figures add daily returns without compounding, and max drawdown is measured on the same summed series. `print_summary()` also reports compounded returns.
- **Costs:** no transaction costs, slippage or financing.
- **Missing data:** missing feature values (such as the VIX proxy gaps above) are passed to LightGBM, which handles them natively. No rows are dropped for missing features; only the final day, which has no next-day target, is excluded.
- **Feature importance:** chart 4 comes from a model fit on the full sample, for description only. It is never used for evaluation.

## Limitations

Results are historical and research-oriented. They are presented to demonstrate the evaluation pipeline and should not be interpreted as evidence of future trading performance or as investment advice. They cover one asset and one data source (OptionsDX end-of-day snapshots at 16:00 ET).

## Tests

```bash
pip install -r requirements.txt
pytest
```

The suite runs in seconds on synthetic options-chain data generated in `tests/conftest.py`, so no purchased data is needed. It checks:

- **Ingest:** bracketed headers, padded values, malformed rows and the liquidity filter.
- **Features:** one row per day, target alignment, and that truncating future data changes no earlier feature.
- **Backtest:** training never reaches the test window, the purge holds, test days are unique and ordered, and runs are deterministic.
- **Known truth:** with no planted signal the win rate stays near 50%, and with a planted signal the pipeline recovers it and beats the baseline.

CI runs the suite on every push.

## Structure

```
├── pipeline/
│   ├── ingest.py      # Load and validate raw options data
│   ├── features.py    # Feature engineering (daily collapse)
│   ├── model.py       # LightGBM classifier: train() and predict()
│   └── backtest.py    # Walk-forward engine, baseline, calibration, regime breakdown
├── tests/
│   ├── conftest.py    # Synthetic options-chain generator with optional planted signal
│   ├── test_ingest.py
│   ├── test_features.py
│   └── test_backtest.py
├── notebooks/
│   └── analysis.ipynb # End-to-end walkthrough with charts
├── outputs/           # Pre-generated charts
├── .github/workflows/tests.yml
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

> **macOS note:** LightGBM requires OpenMP. Install via `brew install libomp` if you see a `libomp.dylib` error on import.

## Data

The pipeline is designed for [OptionsDX](https://www.optionsdx.com) SPX end-of-day option chain data. Each file contains one row per strike/expiry combination with call and put greeks side by side (wide format), snapshotted at 16:00 ET. The data is not included in this repository.

Place the data files under `data/options_dx/` or pass another directory to `load_all(data_dir=...)`. The loader discovers all `.txt` files recursively, so any folder structure works.

## Usage

### Full pipeline

```python
from pipeline.ingest import load_all
from pipeline.features import build_features
from pipeline.backtest import run, print_summary

raw = load_all()
feat = build_features(raw)
results, metrics = run(feat)
print_summary(metrics)  # model vs always long, by regime, calibration
```

### Train and predict independently

```python
import pandas as pd
from pipeline.model import train, predict

model, importance = train(feat, cutoff_date=pd.Timestamp("2022-01-01"))
probs = predict(model, feat[feat["date"] >= pd.Timestamp("2022-01-01")])
signals = (probs >= 0.5).astype(int)  # 1 = predict up, 0 = predict down
```

### Notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

## Features

The pipeline collapses raw strike-level options data to one row per trading date:

| Feature | Description |
|---|---|
| `pc_ratio` | Total put volume / total call volume |
| `atm_pc_ratio` | P/C ratio for ATM strikes (STRIKE_DISTANCE_PCT < 5%) |
| `near_pc_ratio` | P/C ratio for DTE ≤ 30 |
| `far_pc_ratio` | P/C ratio for DTE > 30 |
| `term_skew` | near_pc_ratio − far_pc_ratio |
| `iv_skew` | Mean put IV − mean call IV for ATM strikes |
| `net_delta` | Σ(C_DELTA × C_VOL) + Σ(P_DELTA × P_VOL) |
| `avg_c_spread_pct` | ATM call bid-ask spread as % of midpoint |
| `avg_p_spread_pct` | ATM put bid-ask spread as % of midpoint |
| `vix_proxy` | Mean ATM call IV for DTE 25–35 |
| Rolling features | pc_ratio, iv_skew, net_delta at 5/10/20-day trailing windows |
| `underlying_return_1d` | Daily SPX return |
| `target` | 1 if next day's SPX close is higher, else 0 |

## Charts

![P/C Ratio vs SPX](outputs/pc_ratio_vs_spx.png)
![Walk-Forward P&L](outputs/walkforward_pnl.png)
![Win Rate by Regime](outputs/winrate_by_regime.png)
![Feature Importance](outputs/feature_importance.png)

## License

MIT
