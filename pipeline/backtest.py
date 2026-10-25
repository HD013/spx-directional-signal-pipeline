"""
backtest.py — Walk-forward backtest engine for the SPX directional signal.

Training window : expanding, minimum 504 days (~2 years)
Test window     : 63 days  (~1 quarter)
Step            : 63 days
Purge           : 1 day (the label horizon)

Every fold is scored against an always-long baseline, and out-of-sample
probabilities are checked for calibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.model import train, predict

_TRAIN_DAYS = 504
_TEST_DAYS = 63
_STEP_DAYS = 63
_PURGE_DAYS = 1   # labels look one day ahead

_VIX_LOW = 0.15   # vix_proxy is decimal IV (0.15 = 15%)
_VIX_HIGH = 0.25
_REGIMES = ["low", "medium", "high"]


def _metrics(daily_pnl: pd.Series, correct: pd.Series) -> dict:
    """Compute win rate, Sharpe, max drawdown, and summed/compounded return."""
    if daily_pnl.empty:
        return dict(win_rate=np.nan, sharpe=np.nan, max_drawdown=np.nan,
                    total_return=np.nan, compounded_return=np.nan, n_days=0)

    win_rate = correct.mean()
    total_return = daily_pnl.sum()
    compounded_return = (1 + daily_pnl).prod() - 1
    sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else np.nan

    cum = daily_pnl.cumsum()
    max_drawdown = (cum - cum.cummax()).min()

    return dict(
        win_rate=win_rate,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        total_return=total_return,
        compounded_return=compounded_return,
        n_days=len(daily_pnl),
    )


def _assign_regime(vix_proxy: pd.Series) -> pd.Series:
    """Bucket vix_proxy into low/medium/high. Days with no VIX proxy stay NaN."""
    return pd.cut(
        vix_proxy,
        bins=[-np.inf, _VIX_LOW, _VIX_HIGH, np.inf],
        labels=_REGIMES,
    )


def calibration_table(results: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Reliability table: mean predicted up-probability vs observed up-rate,
    in equal-width probability bins. Empty bins are omitted.
    """
    bins = pd.cut(results["pred_prob"], np.linspace(0, 1, n_bins + 1), include_lowest=True)
    g = results.groupby(bins, observed=True)
    table = pd.DataFrame(
        {
            "mean_predicted": g["pred_prob"].mean(),
            "observed_up_rate": g["actual"].mean(),
            "n_days": g.size(),
        }
    )
    return table.reset_index(names="bin")


def summarize(results: pd.DataFrame) -> dict:
    """
    Compute model, baseline, and calibration metrics from a results DataFrame.

    Parameters
    ----------
    results : pd.DataFrame
        Output of run() (or the cached CSV of it).

    Returns
    -------
    dict
        overall, by_regime      : model metrics
        baseline                : always-long metrics, overall and by regime
        calibration             : Brier score, reference scores, reliability table
        n_no_regime             : OOS days with no VIX proxy (excluded from by_regime)
    """
    regime = _assign_regime(results["vix_proxy"])
    actual_up = results["actual"] == 1

    def _block(pnl: pd.Series, correct: pd.Series) -> dict:
        return {
            "overall": _metrics(pnl, correct),
            "by_regime": {r: _metrics(pnl[regime == r], correct[regime == r]) for r in _REGIMES},
        }

    model = _block(results["daily_pnl"], results["correct"])
    baseline = _block(results["fwd_return"], actual_up)

    up_rate = actual_up.mean()
    calibration = {
        "brier": ((results["pred_prob"] - results["actual"]) ** 2).mean(),
        "brier_coin_flip": 0.25,
        # Constant forecast at the realised OOS up-rate (a hindsight reference)
        "brier_base_rate": up_rate * (1 - up_rate),
        "table": calibration_table(results),
    }

    return {
        "overall": model["overall"],
        "by_regime": model["by_regime"],
        "baseline": baseline,
        "calibration": calibration,
        "n_no_regime": int(regime.isna().sum()),
    }


def run(
    features_df: pd.DataFrame,
    *,
    train_days: int = _TRAIN_DAYS,
    test_days: int = _TEST_DAYS,
    step_days: int = _STEP_DAYS,
    purge_days: int = _PURGE_DAYS,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Run a walk-forward backtest over features_df.

    Each fold trains on all rows before the test window (at least train_days),
    minus the last purge_days rows, whose labels depend on prices inside the
    test window. It then predicts the next test_days rows and steps forward
    step_days.

    Parameters
    ----------
    features_df : pd.DataFrame
        Output of features.build_features().
    train_days, test_days, step_days, purge_days : int
        Walk-forward geometry, in trading days.
    verbose : bool
        Print one line per fold.

    Returns
    -------
    results : pd.DataFrame
        One row per test day: date, pred_prob, predicted, actual, correct,
        vix_proxy, fwd_return, daily_pnl, cumulative_pnl, vix_regime.
    metrics : dict
        Output of summarize(results).
    """
    df = features_df.dropna(subset=["target"]).reset_index(drop=True)
    n = len(df)

    # Forward return: signal on day t earns day t+1's SPX return
    # Kept separate so it is never passed to train/predict as a feature
    forward_return = df["underlying_return_1d"].shift(-1)

    folds = []
    start = train_days
    fold_num = 0

    while start + test_days <= n:
        fold_num += 1
        test_slice = df.iloc[start : start + test_days]
        test_start_date = test_slice["date"].iloc[0]

        # train() keeps rows with date < cutoff, so this drops the purged rows
        cutoff_date = df["date"].iloc[start - purge_days]
        model, _ = train(features_df, cutoff_date=cutoff_date)
        preds = predict(model, test_slice)

        predicted = (preds >= 0.5).astype(int).values
        actual = test_slice["target"].astype(int).values
        fwd_ret = forward_return.iloc[start : start + test_days].fillna(0).values
        signal = 2 * predicted - 1  # long=+1, short=-1

        fold_df = pd.DataFrame(
            {
                "date": test_slice["date"].values,
                "pred_prob": preds.values,
                "predicted": predicted,
                "actual": actual,
                "correct": predicted == actual,
                "vix_proxy": test_slice["vix_proxy"].values,
                "fwd_return": fwd_ret,
                "daily_pnl": signal * fwd_ret,
            }
        )
        folds.append(fold_df)

        if verbose:
            pct = (start - train_days) / (n - train_days - test_days + step_days) * 100
            print(f"  Fold {fold_num:3d}  test start={test_start_date.date()}  "
                  f"win={fold_df['correct'].mean():.0%}  [{pct:.0f}%]")

        start += step_days

    results = pd.concat(folds, ignore_index=True)
    results["cumulative_pnl"] = results["daily_pnl"].cumsum()
    results["vix_regime"] = _assign_regime(results["vix_proxy"])

    return results, summarize(results)


def _print_block(title: str, m: dict, b: dict) -> None:
    """Print one model-vs-baseline block."""
    print(f"\n{title}  n={m['n_days']}")
    print(f"  {'':14s}{'model':>10s}{'always long':>14s}")
    print(f"  {'Win rate':14s}{m['win_rate']:>10.1%}{b['win_rate']:>14.1%}")
    print(f"  {'Sharpe':14s}{m['sharpe']:>10.2f}{b['sharpe']:>14.2f}")
    print(f"  {'Max drawdown':14s}{m['max_drawdown']:>10.1%}{b['max_drawdown']:>14.1%}")
    print(f"  {'Summed return':14s}{m['total_return']:>10.1%}{b['total_return']:>14.1%}")
    print(f"  {'Compounded':14s}{m['compounded_return']:>10.1%}{b['compounded_return']:>14.1%}")


def print_summary(metrics: dict) -> None:
    """Print a formatted performance summary."""
    base = metrics["baseline"]
    _print_block("=== Overall ===", metrics["overall"], base["overall"])

    print("\n=== By VIX Regime ===")
    for regime, m in metrics["by_regime"].items():
        if not m or m["n_days"] == 0:
            continue
        _print_block(f"[{regime.upper()} VIX]", m, base["by_regime"][regime])
    print(f"\n  Days with no VIX proxy (excluded above): {metrics['n_no_regime']}")

    cal = metrics["calibration"]
    print("\n=== Calibration ===")
    print(f"  Brier score            : {cal['brier']:.4f}")
    print(f"  Coin flip (0.5)        : {cal['brier_coin_flip']:.4f}")
    print(f"  Constant at OOS up-rate: {cal['brier_base_rate']:.4f}")


if __name__ == "__main__":
    from pipeline.ingest import load_all
    from pipeline.features import build_features

    print("Loading data...")
    raw = load_all(verbose=False)
    print("Building features...")
    feat = build_features(raw)

    print(f"Running walk-forward backtest (expanding window, min {_TRAIN_DAYS}d / "
          f"{_TEST_DAYS}d test / {_STEP_DAYS}d step / {_PURGE_DAYS}d purge)...")
    results, metrics = run(feat)

    print_summary(metrics)
    print(f"\nResults shape : {results.shape}")
    print(f"Date range    : {results['date'].min().date()} → {results['date'].max().date()}")
