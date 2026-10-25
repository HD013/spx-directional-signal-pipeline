"""
features.py — Feature engineering for the SPX directional signal pipeline.

Collapses the strike-level DataFrame (output of ingest.load_all) to one row
per trading date and engineers options-flow and volatility features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_ATM_PCT = 0.05       # STRIKE_DISTANCE_PCT threshold for ATM classification
_NEAR_DTE = 30        # DTE boundary between near- and far-term
_VIX_DTE_LO = 25     # Lower DTE bound for VIX proxy
_VIX_DTE_HI = 35     # Upper DTE bound for VIX proxy
_ROLL_WINDOWS = [5, 10, 20]


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Divide num by den, returning NaN where den is zero."""
    return num / den.replace(0, np.nan)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse strike-level options data to one row per trading date.

    Parameters
    ----------
    df : pd.DataFrame
        Clean output of ingest.load_all().

    Returns
    -------
    pd.DataFrame
        Daily feature DataFrame with a 'date' column and binary 'target'.
        The last row has target=NaN (no future price available).
    """
    # Boolean masks computed once and reused
    atm = df["STRIKE_DISTANCE_PCT"] < _ATM_PCT
    near = df["DTE"] <= _NEAR_DTE
    vix_win = atm & df["DTE"].between(_VIX_DTE_LO, _VIX_DTE_HI)

    date_col = df["QUOTE_DATE"]

    # --- Full-chain groupbys ---
    g = df.groupby(date_col)
    g_atm = df[atm].groupby(date_col)
    g_near = df[near].groupby(date_col)
    g_far = df[~near].groupby(date_col)
    g_vix = df[vix_win].groupby(date_col)

    # P/C ratios
    pc_ratio = _safe_div(g["P_VOLUME"].sum(), g["C_VOLUME"].sum())
    atm_pc_ratio = _safe_div(g_atm["P_VOLUME"].sum(), g_atm["C_VOLUME"].sum())
    near_pc_ratio = _safe_div(g_near["P_VOLUME"].sum(), g_near["C_VOLUME"].sum())
    far_pc_ratio = _safe_div(g_far["P_VOLUME"].sum(), g_far["C_VOLUME"].sum())

    # IV skew: ATM put IV - ATM call IV
    iv_skew = g_atm["P_IV"].mean() - g_atm["C_IV"].mean()

    # Net delta: volume-weighted call delta + put delta (puts have negative delta)
    c_delta_vol = (df["C_DELTA"] * df["C_VOLUME"].fillna(0)).groupby(date_col).sum()
    p_delta_vol = (df["P_DELTA"] * df["P_VOLUME"].fillna(0)).groupby(date_col).sum()
    net_delta = c_delta_vol + p_delta_vol

    # Bid-ask spread pct for ATM options
    atm_df = df[atm]
    c_mid = (atm_df["C_BID"] + atm_df["C_ASK"]) / 2
    p_mid = (atm_df["P_BID"] + atm_df["P_ASK"]) / 2
    c_spread = (atm_df["C_ASK"] - atm_df["C_BID"]) / c_mid.replace(0, np.nan)
    p_spread = (atm_df["P_ASK"] - atm_df["P_BID"]) / p_mid.replace(0, np.nan)
    avg_c_spread_pct = c_spread.groupby(atm_df["QUOTE_DATE"]).mean()
    avg_p_spread_pct = p_spread.groupby(atm_df["QUOTE_DATE"]).mean()

    # VIX proxy: mean ATM call IV in the 25-35 DTE window
    vix_proxy = g_vix["C_IV"].mean()

    # Underlying close price (all rows per date share the same EOD snapshot)
    underlying = g["UNDERLYING_LAST"].first()

    # --- Assemble daily DataFrame ---
    feat = pd.DataFrame(
        {
            "underlying": underlying,
            "pc_ratio": pc_ratio,
            "atm_pc_ratio": atm_pc_ratio,
            "near_pc_ratio": near_pc_ratio,
            "far_pc_ratio": far_pc_ratio,
            "iv_skew": iv_skew,
            "net_delta": net_delta,
            "avg_c_spread_pct": avg_c_spread_pct,
            "avg_p_spread_pct": avg_p_spread_pct,
            "vix_proxy": vix_proxy,
        },
        index=underlying.index,
    )

    feat["term_skew"] = feat["near_pc_ratio"] - feat["far_pc_ratio"]

    # --- Rolling windows for pc_ratio, iv_skew, net_delta ---
    for col in ["pc_ratio", "iv_skew", "net_delta"]:
        for w in _ROLL_WINDOWS:
            feat[f"{col}_roll{w}"] = feat[col].rolling(w, min_periods=w).mean()

    # --- Underlying daily return ---
    feat["underlying_return_1d"] = feat["underlying"].pct_change()

    # --- Target: 1 if next trading day's close is higher, else 0 ---
    next_price = feat["underlying"].shift(-1)
    feat["target"] = (next_price > feat["underlying"]).astype("float32")
    feat.loc[next_price.isna(), "target"] = np.nan  # last row has no forward price

    feat = feat.drop(columns=["underlying"])
    feat.index.name = "date"
    feat = feat.reset_index()

    return feat


if __name__ == "__main__":
    from pipeline.ingest import load_all

    print("Loading raw data...")
    raw = load_all(verbose=False)

    print("Engineering features...")
    features = build_features(raw)

    print(f"\nFeature DataFrame shape : {features.shape}")
    print(f"Date range              : {features['date'].min().date()} → {features['date'].max().date()}")
    print(f"Trading days            : {len(features)}")
    print(f"Target distribution     : {features['target'].value_counts().to_dict()}")
    print(f"NaN counts (top 5):\n{features.isnull().sum().sort_values(ascending=False).head()}")
    print(f"\nSample (last 5 rows):\n{features.tail().to_string()}")
