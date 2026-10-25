"""
Tests for pipeline.backtest: fold boundaries, purge, and known-truth checks.

Uses small walk-forward windows so the suite runs in seconds.
"""

from __future__ import annotations

import pandas as pd
import pytest

import pipeline.backtest as bt
from pipeline.features import build_features

WINDOWS = dict(train_days=200, test_days=50, step_days=50, verbose=False)


@pytest.fixture
def null_feat(chain):
    return build_features(chain(signal_strength=0.0, seed=1))


@pytest.fixture
def planted_feat(chain):
    return build_features(chain(signal_strength=1.0, seed=2))


def test_folds_never_see_test_period(null_feat, monkeypatch):
    calls = []
    real_train = bt.train

    def recording_train(features_df, cutoff_date=None):
        used = features_df[(features_df["date"] < cutoff_date) & features_df["target"].notna()]
        calls.append(used["date"].max())
        return real_train(features_df, cutoff_date=cutoff_date)

    monkeypatch.setattr(bt, "train", recording_train)
    results, _ = bt.run(null_feat, **WINDOWS)

    dates = list(null_feat.dropna(subset=["target"])["date"])
    test_starts = results["date"].iloc[:: WINDOWS["test_days"]].tolist()
    assert len(calls) == len(test_starts)

    for train_max, test_start in zip(calls, test_starts):
        assert train_max < test_start
        # Exactly purge_days trading days sit between training and testing
        gap = dates.index(test_start) - dates.index(train_max) - 1
        assert gap == bt._PURGE_DAYS


def test_test_days_are_unique_and_ordered(null_feat):
    results, _ = bt.run(null_feat, **WINDOWS)
    assert results["date"].is_unique and results["date"].is_monotonic_increasing
    assert len(results) % WINDOWS["test_days"] == 0


def test_no_signal_gives_coin_flip(null_feat):
    _, metrics = bt.run(null_feat, **WINDOWS)
    assert 0.40 <= metrics["overall"]["win_rate"] <= 0.60


def test_planted_signal_is_recovered(planted_feat):
    _, metrics = bt.run(planted_feat, **WINDOWS)
    assert metrics["overall"]["win_rate"] > 0.60
    assert metrics["overall"]["win_rate"] > metrics["baseline"]["overall"]["win_rate"]


def test_runs_are_deterministic(null_feat):
    a, _ = bt.run(null_feat, **WINDOWS)
    b, _ = bt.run(null_feat, **WINDOWS)
    pd.testing.assert_frame_equal(a, b)


def test_summary_has_baseline_and_calibration(null_feat):
    results, metrics = bt.run(null_feat, **WINDOWS)

    assert set(metrics["baseline"]) == {"overall", "by_regime"}
    cal = metrics["calibration"]
    assert 0 <= cal["brier"] <= 1
    assert cal["table"]["n_days"].sum() == len(results)
    assert metrics["n_no_regime"] == results["vix_proxy"].isna().sum()

    # summarize() on the cached form of results reproduces run()'s metrics
    assert bt.summarize(results)["overall"] == metrics["overall"]
