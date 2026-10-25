"""
model.py — LightGBM binary classifier for SPX directional prediction.

Exposes train() and predict() as standalone importable functions.
The walk-forward engine in backtest.py calls these repeatedly across
rolling windows; this module owns no state between calls.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

# Columns that are never model features
_NON_FEATURE_COLS = {"date", "target"}

_LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "importance_type": "gain",  # affects reported importances only, not predictions
}


def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the sorted list of feature column names for a given DataFrame."""
    return [c for c in df.columns if c not in _NON_FEATURE_COLS]


def train(
    features_df: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
) -> tuple[lgb.LGBMClassifier, pd.DataFrame]:
    """
    Fit a LightGBM binary classifier on features_df up to cutoff_date.

    Parameters
    ----------
    features_df : pd.DataFrame
        Output of features.build_features(). Must contain 'date' and 'target'.
    cutoff_date : pd.Timestamp, optional
        Train on rows where date < cutoff_date. If None, uses all rows.

    Returns
    -------
    model : lgb.LGBMClassifier
        Fitted classifier.
    importance : pd.DataFrame
        Feature importances with columns ['feature', 'importance'], sorted
        descending by importance.
    """
    df = features_df.copy()
    df = df[df["target"].notna()]

    if cutoff_date is not None:
        df = df[df["date"] < cutoff_date]

    if df.empty:
        raise ValueError("No training rows after applying cutoff_date and dropping NaN targets.")

    feat_cols = _feature_cols(df)
    X = df[feat_cols]
    y = df["target"].astype(int)

    model = lgb.LGBMClassifier(**_LGB_PARAMS)
    model.fit(X, y)

    importance = pd.DataFrame(
        {"feature": feat_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    return model, importance


def predict(
    model: lgb.LGBMClassifier,
    features_df: pd.DataFrame,
) -> pd.Series:
    """
    Generate predicted up-day probabilities for features_df.

    Parameters
    ----------
    model : lgb.LGBMClassifier
        Fitted model returned by train().
    features_df : pd.DataFrame
        Feature rows to score. Must contain the same feature columns used
        during training.

    Returns
    -------
    pd.Series
        Probability of an up day (class 1), indexed by features_df.index.
    """
    feat_cols = _feature_cols(features_df)
    X = features_df[feat_cols]
    probs = model.predict_proba(X)[:, 1]
    return pd.Series(probs, index=features_df.index, name="pred_prob")


if __name__ == "__main__":
    from pipeline.ingest import load_all
    from pipeline.features import build_features

    print("Loading data...")
    raw = load_all(verbose=False)
    print("Building features...")
    feat = build_features(raw)

    # Train on first 8 years, score the next year
    train_end = pd.Timestamp("2018-01-01")
    test_mask = (feat["date"] >= train_end) & (feat["date"] < pd.Timestamp("2019-01-01"))

    print(f"Training on rows before {train_end.date()}...")
    model, importance = train(feat, cutoff_date=train_end)

    print(f"Top 10 features:\n{importance.head(10).to_string(index=False)}")

    test_df = feat[test_mask].reset_index(drop=True)
    preds = predict(model, test_df)

    signals = (preds >= 0.5).astype(int)
    actual = test_df["target"].astype(int)
    win_rate = (signals == actual).mean()
    print(f"\nTest period : {test_df['date'].min().date()} → {test_df['date'].max().date()}")
    print(f"Test rows   : {len(test_df)}")
    print(f"Win rate    : {win_rate:.1%}")
