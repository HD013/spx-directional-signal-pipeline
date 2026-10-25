"""Tests for pipeline.features: daily collapse, target alignment, causality."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.features import build_features


def test_one_row_per_date(chain):
    raw = chain(n_days=120)
    feat = build_features(raw)
    assert len(feat) == raw["QUOTE_DATE"].nunique()
    assert feat["date"].is_unique and feat["date"].is_monotonic_increasing


def test_target_is_next_day_direction(chain):
    raw = chain(n_days=120)
    feat = build_features(raw)
    price = raw.groupby("QUOTE_DATE")["UNDERLYING_LAST"].first().values
    expected = (price[1:] > price[:-1]).astype("float32")

    np.testing.assert_array_equal(feat["target"].values[:-1], expected)
    assert np.isnan(feat["target"].iloc[-1])  # no forward price for the last day


def test_features_use_only_past_data(chain):
    """Truncating the data at date D must not change any feature on or before D."""
    raw = chain(n_days=200)
    full = build_features(raw)
    cut = full["date"].iloc[150]
    trunc = build_features(raw[raw["QUOTE_DATE"] <= cut])

    before = full["date"] < cut
    pd.testing.assert_frame_equal(
        full[before].reset_index(drop=True),
        trunc[trunc["date"] < cut].reset_index(drop=True),
    )
    # On day D itself every feature matches; only the target is unknown
    feat_cols = [c for c in full.columns if c != "target"]
    pd.testing.assert_frame_equal(
        full.loc[full["date"] == cut, feat_cols].reset_index(drop=True),
        trunc.loc[trunc["date"] == cut, feat_cols].reset_index(drop=True),
    )
