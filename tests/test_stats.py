"""Tests for pipeline.stats: each check must be able to fail as well as pass."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import stats


def _blocked(mean: float, n_blocks: int = 60, per_block: int = 20, seed: int = 0):
    """Values with a shared shock per block, so rows within a block co-move."""
    rng = np.random.default_rng(seed)
    blocks = np.repeat(np.arange(n_blocks), per_block)
    values = mean + rng.normal(0, 1, n_blocks)[blocks] + rng.normal(0, 1, len(blocks))
    return values, blocks


def test_month_block_is_consecutive_across_years():
    m = stats.month_block(pd.Series(pd.to_datetime(["2019-12-31", "2020-01-02"])))
    assert m.iloc[1] - m.iloc[0] == 1


def test_bootstrap_includes_zero_for_null_and_excludes_it_for_signal():
    null = stats.block_bootstrap_mean(*_blocked(0.0))
    planted = stats.block_bootstrap_mean(*_blocked(1.0))
    assert null["lo"] < 0 < null["hi"]
    assert planted["lo"] > 0


def test_bootstrap_is_wider_than_a_row_level_interval():
    values, blocks = _blocked(0.0)
    block_ci = stats.block_bootstrap_mean(values, blocks)
    row_ci = stats.block_bootstrap_mean(values, np.arange(len(values)))
    assert (block_ci["hi"] - block_ci["lo"]) > 2 * (row_ci["hi"] - row_ci["lo"])


def test_bootstrap_flags_thin_block_counts():
    values, blocks = _blocked(0.0, n_blocks=6)
    assert stats.block_bootstrap_mean(values, blocks)["thin"]
    with pytest.raises(ValueError):
        stats.block_bootstrap_mean([1.0, 2.0], [0, 0])


def test_episodes_counts_runs_of_consecutive_blocks():
    assert stats.episodes([1, 2, 3, 5, 6, 9]) == 3
    assert stats.episodes([4, 4, 4]) == 1
    assert stats.episodes([]) == 0


def test_breakeven_cost_zeroes_net_return():
    rng = np.random.default_rng(1)
    ret = rng.normal(0.001, 0.01, 500)
    pos = np.sign(ret + rng.normal(0, 0.01, 500))  # informative, trades often
    out = stats.turnover_and_breakeven(pos, ret)
    turnover = np.abs(np.diff(pos, prepend=0.0))
    net = (pos * ret - out["breakeven"] * turnover).sum()
    assert out["breakeven"] > 0
    assert abs(net) < 1e-12


def test_relative_breakeven_against_a_buy_and_hold_baseline():
    ret = np.array([0.01, -0.02, 0.03, -0.01])
    model = np.array([1.0, -1.0, 1.0, -1.0])   # perfect foresight, flips every day
    hold = np.ones(4)
    rel = stats.relative_breakeven(model, hold, ret)
    m = stats.turnover_and_breakeven(model, ret)
    h = stats.turnover_and_breakeven(hold, ret)
    # At the relative breakeven the two net returns are equal
    assert np.isclose(m["gross_total"] - rel * m["turnover_total"],
                      h["gross_total"] - rel * h["turnover_total"])
    assert np.isnan(stats.relative_breakeven(hold, hold, ret))


def test_brier_decomposition_identity_is_exact_for_binned_forecasts():
    rng = np.random.default_rng(2)
    p = rng.choice([0.2, 0.4, 0.6, 0.8], 4000)
    y = (rng.uniform(size=4000) < p).astype(float)
    # Equal-width edges put each distinct value in its own bin (quantile edges
    # would land on the tied values and merge two of them)
    d = stats.brier_decomposition(y, p, n_bins=4, scheme="equal_width")
    assert abs(d["residual"]) < 1e-12
    assert d["resolution"] > 0.02


def test_constant_forecast_has_no_resolution():
    y = np.random.default_rng(3).integers(0, 2, 1000).astype(float)
    d = stats.brier_decomposition(y, np.full(1000, y.mean()), n_bins=5)
    assert d["resolution"] == pytest.approx(0.0, abs=1e-12)


def test_calibration_slopes_separate_calibrated_from_uninformative():
    rng = np.random.default_rng(4)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.uniform(size=20000) < p).astype(float)
    good = stats.calibration_slopes(y, p)
    noise = stats.calibration_slopes(y, rng.uniform(0.05, 0.95, 20000))
    assert len(good) == 6
    assert good["slope"].between(0.9, 1.1).all()
    assert noise["slope"].abs().max() < 0.2


def test_decay_detects_a_fading_series():
    fading = np.r_[np.full(10, 0.6), np.full(10, 0.5)] + np.random.default_rng(5).normal(0, 0.01, 20)
    d = stats.decay(fading)
    assert d["difference"] < -5 * d["se"]
    flat = stats.decay(np.random.default_rng(6).normal(0.5, 0.01, 20))
    assert abs(flat["difference"]) < 3 * flat["se"]
