"""
Shared fixtures: a synthetic options chain in the post-ingest schema.

The generator plants a known relationship (or none) between today's ATM
put-minus-call IV and tomorrow's underlying return, so tests can check that
the pipeline finds signal when it exists and does not invent it when it
does not. No purchased data is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make `pipeline` importable when pytest is run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCHEMA = [
    "QUOTE_UNIXTIME", "QUOTE_READTIME", "QUOTE_DATE", "QUOTE_TIME_HOURS", "UNDERLYING_LAST",
    "EXPIRE_DATE", "EXPIRE_UNIX", "DTE", "C_DELTA", "C_GAMMA", "C_VEGA", "C_THETA", "C_RHO", "C_IV",
    "C_VOLUME", "C_LAST", "C_SIZE", "C_BID", "C_ASK", "STRIKE", "P_BID", "P_ASK", "P_SIZE", "P_LAST",
    "P_DELTA", "P_GAMMA", "P_VEGA", "P_THETA", "P_RHO", "P_IV", "P_VOLUME", "STRIKE_DISTANCE",
    "STRIKE_DISTANCE_PCT",
]

_DTES = [7, 30, 60]                 # near, VIX-window, far
_STRIKE_PCTS = [0.0, 0.02, 0.08]    # two ATM strikes, one OTM


def make_chain(n_days: int = 650, signal_strength: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """
    Build a synthetic strike-level chain shaped like ingest.load_all() output.

    Next-day return = signal_strength * skew_t + noise, where skew_t is the
    ATM put IV minus call IV on day t. signal_strength=0 gives no signal.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-02", periods=n_days)

    skew = rng.normal(0, 0.02, n_days)
    noise = rng.normal(0, 0.01, n_days)
    ret = np.zeros(n_days)
    ret[1:] = signal_strength * skew[:-1] + noise[1:]
    price = 2000 * np.cumprod(1 + ret)
    base_iv = 0.18 + 0.05 * np.abs(np.sin(np.arange(n_days) / 60))

    grid = [(d, p) for d in _DTES for p in _STRIKE_PCTS]
    rows_per_day = len(grid)
    day = np.repeat(np.arange(n_days), rows_per_day)
    dte = np.tile([g[0] for g in grid], n_days).astype("float32")
    pct = np.tile([g[1] for g in grid], n_days).astype("float32")
    n = len(day)

    df = pd.DataFrame(
        {
            "QUOTE_DATE": dates[day],
            "UNDERLYING_LAST": price[day].astype("float32"),
            "DTE": dte,
            "STRIKE_DISTANCE_PCT": pct,
            "STRIKE": (price[day] * (1 + pct)).astype("float32"),
            "C_IV": (base_iv[day] - skew[day] / 2).astype("float32"),
            "P_IV": (base_iv[day] + skew[day] / 2).astype("float32"),
            "C_VOLUME": rng.poisson(500, n).astype("float32"),
            "P_VOLUME": rng.poisson(600, n).astype("float32"),
            "C_DELTA": (0.5 - pct * 3).astype("float32"),
            "P_DELTA": (-0.5 + pct * 3).astype("float32"),
            "C_BID": rng.uniform(9, 10, n).astype("float32"),
            "P_BID": rng.uniform(9, 10, n).astype("float32"),
        }
    )
    df["C_ASK"] = df["C_BID"] + 0.5
    df["P_ASK"] = df["P_BID"] + 0.5
    df["EXPIRE_DATE"] = df["QUOTE_DATE"] + pd.to_timedelta(df["DTE"], unit="D")

    for col in SCHEMA:
        if col not in df.columns:
            df[col] = np.nan
    return df[SCHEMA]


@pytest.fixture
def chain():
    """Factory fixture: chain(n_days=..., signal_strength=..., seed=...)."""
    return make_chain
