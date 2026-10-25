"""Tests for pipeline.ingest: OptionsDX file quirks and the liquidity filter."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import SCHEMA
from pipeline.ingest import load_all


def _row(**overrides) -> list[str]:
    base = {c: "1" for c in SCHEMA}
    base.update(
        QUOTE_READTIME="2020-01-02 16:00",
        QUOTE_DATE="2020-01-02",
        EXPIRE_DATE="2020-01-31",
        C_SIZE="101 x 101",
        P_SIZE="5 x 7",
        STRIKE="3200",
        C_VOLUME="10",
        P_VOLUME="5",
    )
    base.update(overrides)
    return [base[c] for c in SCHEMA]


def _write(path, rows: list[list[str]]) -> None:
    header = ", ".join(f"[{c}]" for c in SCHEMA)
    lines = [header] + [", ".join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n")


def test_load_handles_optionsdx_quirks(tmp_path):
    sub = tmp_path / "spx_eod_2020-abc123"
    sub.mkdir()
    _write(
        sub / "spx_eod_202001.txt",
        [
            _row(),                                   # liquid
            _row(C_VOLUME="", P_VOLUME="3"),          # blank call volume, puts traded
            _row(C_VOLUME="0", P_VOLUME=""),          # illiquid: dropped
            _row() + ["extra", "fields"],             # malformed: skipped
        ],
    )

    df = load_all(tmp_path, verbose=False)

    assert list(df.columns) == SCHEMA                 # brackets stripped
    assert len(df) == 2
    assert df["C_VOLUME"].isna().sum() == 1
    assert df["STRIKE"].dtype == "float32"
    assert pd.api.types.is_datetime64_any_dtype(df["QUOTE_DATE"])
    assert df["C_SIZE"].iloc[0] == "101 x 101"        # left as a string


def test_empty_directory_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_all(tmp_path, verbose=False)
