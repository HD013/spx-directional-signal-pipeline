"""
stats.py — Statistical checks that qualify the walk-forward results.

Standard methods, kept small and dependency-free (numpy only):

block_bootstrap_mean   interval on a mean, resampling whole calendar months
episodes               number of separate runs of consecutive months
turnover_and_breakeven trading cost at which a position series stops making money
relative_breakeven     cost at which a model stops beating a baseline
brier_decomposition    Brier score split into reliability, resolution, uncertainty
calibration_slopes     reliability slope under several binnings, with the spread
decay                  first-half vs second-half comparison of per-fold scores

Daily returns are not independent of each other in every respect (volatility
clusters, regimes persist), so intervals resample months rather than days.
Months also give a natural unit for counting how many separate episodes a
regime actually contains.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_THIN_BLOCKS = 12   # fewer blocks than this and a bootstrap interval is optimistic


def month_block(dates: pd.Series) -> pd.Series:
    """Integer month index (year * 12 + month), so consecutive months differ by 1."""
    d = pd.to_datetime(dates)
    return (d.dt.year * 12 + d.dt.month).astype(int)


def block_bootstrap_mean(
    values,
    blocks,
    level: float = 0.90,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict:
    """
    Percentile interval on the mean of `values`, resampling whole blocks.

    Each resample draws blocks with replacement and takes the mean over all
    rows of the drawn blocks. Resampling days instead would treat days within
    the same month as independent and give an interval that is too narrow
    whenever they are not.

    Returns estimate, lo, hi, level, n_blocks, and thin (True when there are
    fewer than 12 blocks, in which case the interval is optimistic).
    """
    v = np.asarray(values, dtype=float)
    _, inv = np.unique(np.asarray(blocks), return_inverse=True)
    sums = np.bincount(inv, weights=v)
    sizes = np.bincount(inv)
    n = len(sums)
    if n < 2:
        raise ValueError(f"need at least 2 blocks to resample, got {n}")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    draws = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    a = (1 - level) / 2
    return dict(
        estimate=float(v.mean()),
        lo=float(np.quantile(draws, a)),
        hi=float(np.quantile(draws, 1 - a)),
        level=level,
        n_blocks=n,
        thin=n < _THIN_BLOCKS,
    )


def episodes(blocks) -> int:
    """
    Number of maximal runs of consecutive blocks.

    Ten high-volatility months in one stretch are one episode, not ten
    independent observations. The episode count is the honest sample size for
    a regime that arrives in runs.
    """
    b = np.unique(np.asarray(blocks))
    if len(b) == 0:
        return 0
    return int(1 + (np.diff(b) != 1).sum())


def turnover_and_breakeven(position, ret) -> dict:
    """
    Turnover of a position series and the cost at which its total return is zero.

    `position` is held over each row and earns that row's `ret`; rows must be
    in time order. The position before the first row is flat, so entering it
    counts as a trade. With a linear cost c per unit of position change,
    net = gross - c * turnover, so the breakeven is exact: gross / turnover.
    Costs are in the units of `ret` (multiply by 1e4 for basis points).
    """
    pos = np.asarray(position, dtype=float)
    r = np.asarray(ret, dtype=float)
    turnover = np.abs(np.diff(pos, prepend=0.0))
    gross = float((pos * r).sum())
    total_turnover = float(turnover.sum())
    return dict(
        gross_total=gross,
        turnover_total=total_turnover,
        turnover_per_row=total_turnover / len(pos),
        breakeven=gross / total_turnover if total_turnover > 0 else np.nan,
    )


def relative_breakeven(position, baseline_position, ret) -> float:
    """
    Cost per unit of position change above which the model's net return falls
    below the baseline's. NaN if the model does not trade more than the baseline.
    """
    m = turnover_and_breakeven(position, ret)
    b = turnover_and_breakeven(baseline_position, ret)
    extra_turnover = m["turnover_total"] - b["turnover_total"]
    if extra_turnover <= 0:
        return np.nan
    return (m["gross_total"] - b["gross_total"]) / extra_turnover


def _bin_index(p: np.ndarray, n_bins: int, scheme: str) -> np.ndarray:
    """Assign each prediction to a bin: equal_count (quantile edges) or equal_width."""
    if scheme == "equal_count":
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    elif scheme == "equal_width":
        edges = np.linspace(p.min(), p.max(), n_bins + 1)
    else:
        raise ValueError(f"unknown binning scheme: {scheme!r}")
    return np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)


def _bin_table(y: np.ndarray, p: np.ndarray, n_bins: int, scheme: str) -> pd.DataFrame:
    idx = _bin_index(p, n_bins, scheme)
    df = pd.DataFrame({"bin": idx, "p": p, "y": y})
    return df.groupby("bin").agg(n=("y", "size"), p_mean=("p", "mean"), y_mean=("y", "mean"))


def brier_decomposition(y, p, n_bins: int = 10, scheme: str = "equal_count") -> dict:
    """
    Murphy decomposition: brier ≈ reliability - resolution + uncertainty.

    reliability  cost of predicted probabilities not matching observed rates (lower is better)
    resolution   credit for separating up days from down days (higher is better)
    uncertainty  variance of the outcome itself; belongs to the problem, not the model

    The identity is exact when predictions are constant within each bin;
    otherwise the small within-bin remainder is returned as `residual`.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    b = _bin_table(y, p, n_bins, scheme)
    n = len(y)
    base = y.mean()
    reliability = float((b["n"] * (b["p_mean"] - b["y_mean"]) ** 2).sum() / n)
    resolution = float((b["n"] * (b["y_mean"] - base) ** 2).sum() / n)
    uncertainty = float(base * (1 - base))
    brier = float(((p - y) ** 2).mean())
    return dict(
        brier=brier,
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        residual=brier - (reliability - resolution + uncertainty),
    )


def calibration_slopes(
    y,
    p,
    schemes=("equal_count", "equal_width"),
    bin_counts=(5, 10, 20),
    min_bin: int = 20,
) -> pd.DataFrame:
    """
    Slope of observed up-rate on mean predicted probability, per binning.

    A slope of 1 means predictions move one for one with outcomes; near 0 means
    they carry little information. Binning is a free choice that can move the
    slope, so every combination is reported rather than one picked. Bins with
    fewer than `min_bin` rows are left out of the fit.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    rows = []
    for scheme in schemes:
        for k in bin_counts:
            b = _bin_table(y, p, k, scheme)
            fit = b[b["n"] >= min_bin]
            slope = np.polyfit(fit["p_mean"], fit["y_mean"], 1)[0] if len(fit) >= 2 else np.nan
            rows.append({"scheme": scheme, "n_bins": k, "bins_used": len(fit), "slope": float(slope)})
    return pd.DataFrame(rows)


def decay(fold_values) -> dict:
    """
    Compare the first and second halves of a sequence of per-fold scores.

    Returns both half-means, their difference (second minus first), and the
    standard error of that difference, so a drop can be read against noise.
    With an odd count the middle fold goes to the second half.
    """
    v = np.asarray(fold_values, dtype=float)
    half = len(v) // 2
    first, second = v[:half], v[half:]
    se = np.sqrt(first.var(ddof=1) / len(first) + second.var(ddof=1) / len(second))
    return dict(
        first_half=float(first.mean()),
        second_half=float(second.mean()),
        difference=float(second.mean() - first.mean()),
        se=float(se),
        n_folds=len(v),
    )
