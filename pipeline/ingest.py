"""
ingest.py — Load and validate all OptionsDX SPX EOD option chain files.

Discovers all monthly .txt files recursively under DATA_DIR, normalises column
names, casts types for memory efficiency, and filters out illiquid strikes
where both call and put volume are zero or null.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Project root is one level above this file (intro_project/)
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "options_dx"

# Columns to cast to float32 (saves ~50% memory vs float64)
_FLOAT32_COLS = [
    "QUOTE_TIME_HOURS",
    "UNDERLYING_LAST",
    "DTE",
    "C_DELTA", "C_GAMMA", "C_VEGA", "C_THETA", "C_RHO", "C_IV",
    "C_VOLUME", "C_LAST", "C_BID", "C_ASK",
    "STRIKE",
    "P_BID", "P_ASK", "P_LAST", "P_DELTA", "P_GAMMA", "P_VEGA",
    "P_THETA", "P_RHO", "P_IV", "P_VOLUME",
    "STRIKE_DISTANCE", "STRIKE_DISTANCE_PCT",
]


def _discover_files(data_dir: Path) -> list[Path]:
    """Return all .txt files under data_dir, sorted by path."""
    return sorted(data_dir.rglob("*.txt"))


def _load_file(path: Path) -> pd.DataFrame:
    """
    Read a single OptionsDX .txt file into a DataFrame.

    Handles bracket-wrapped column names (e.g. [QUOTE_DATE] -> QUOTE_DATE)
    and space-padded values.
    """
    df = pd.read_csv(
        path,
        skipinitialspace=True,
        low_memory=False,
        na_values=["", " "],
        on_bad_lines="skip",
    )
    # Strip [bracket] wrappers from every column name
    df.columns = [c.strip().strip("[]").strip() for c in df.columns]
    return df


def _cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast numeric columns to float32 and parse date columns.

    Coerces unparseable values to NaN rather than raising.
    """
    for col in _FLOAT32_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"], errors="coerce")
    df["EXPIRE_DATE"] = pd.to_datetime(df["EXPIRE_DATE"], errors="coerce")
    return df


def load_all(data_dir: Path = DATA_DIR, verbose: bool = True) -> pd.DataFrame:
    """
    Load, clean, and concatenate all OptionsDX SPX EOD files.

    Drops rows where both C_VOLUME and P_VOLUME are zero or null — these are
    illiquid deep-OTM strikes that carry no signal.

    Parameters
    ----------
    data_dir : Path
        Root directory containing the .txt files (searched recursively).
    verbose : bool
        Print a load summary to stdout.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame sorted by QUOTE_DATE, one row per strike/expiry/date.
    """
    files = _discover_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No .txt files found under {data_dir}")

    chunks: list[pd.DataFrame] = []
    for path in files:
        df = _load_file(path)
        df = _cast_types(df)
        chunks.append(df)

    raw = pd.concat(chunks, ignore_index=True)
    total_raw = len(raw)

    # Keep rows where at least one side traded
    c_vol = raw["C_VOLUME"].fillna(0)
    p_vol = raw["P_VOLUME"].fillna(0)
    liquid_mask = (c_vol > 0) | (p_vol > 0)
    clean = raw[liquid_mask].copy()
    rows_dropped = total_raw - len(clean)

    clean = clean.sort_values("QUOTE_DATE").reset_index(drop=True)

    if verbose:
        print(f"Files loaded     : {len(files)}")
        print(f"Date range       : {clean['QUOTE_DATE'].min().date()} → {clean['QUOTE_DATE'].max().date()}")
        print(f"Total rows (raw) : {total_raw:,}")
        print(f"Rows dropped     : {rows_dropped:,}  ({rows_dropped / total_raw:.1%} illiquid)")
        print(f"Rows retained    : {len(clean):,}")
        print(f"Unique expiries  : {clean['EXPIRE_DATE'].nunique():,}")
        print(f"DataFrame shape  : {clean.shape}")

    return clean


if __name__ == "__main__":
    df = load_all()
