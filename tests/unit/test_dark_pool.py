"""Unit tests for the FINRA dark-pool / short-volume data provider."""

from __future__ import annotations

import pandas as pd

from src.data.alternative import DarkPoolDataProvider
from src.data.alternative.dark_pool import _DARK_POOL_SCHEMA

# A representative FINRA CNMSshvol daily file body (header + data + footer).
_SAMPLE = (
    "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
    "20240115|AAPL|12000000|3000|30000000|B,Q,N,D\n"
    "20240115|MSFT|5000000|1000|20000000|B,Q,N,D\n"
    "20240115|TSLA|0|0|0|B,Q,N,D\n"  # zero total volume -> ratio 0, no divide-by-zero
    "Trade Date|0123456789\n"  # footer line, must be ignored
)


def test_parse_extracts_rows_and_ratio() -> None:
    df = DarkPoolDataProvider._parse(_SAMPLE)
    assert len(df) == 3
    aapl = df[df["symbol"] == "AAPL"].iloc[0]
    assert aapl["short_volume"] == 12_000_000
    assert aapl["total_volume"] == 30_000_000
    # 12M / 30M = 0.4
    assert abs(aapl["short_volume_ratio"] - 0.4) < 1e-9
    assert aapl["source"] == "finra"


def test_parse_handles_zero_total_volume() -> None:
    df = DarkPoolDataProvider._parse(_SAMPLE)
    tsla = df[df["symbol"] == "TSLA"].iloc[0]
    assert tsla["short_volume_ratio"] == 0.0


def test_parse_ignores_header_and_footer() -> None:
    df = DarkPoolDataProvider._parse(_SAMPLE)
    # No row should have a non-numeric / header date.
    assert "Date" not in set(df["symbol"])
    assert all(isinstance(d, pd.Timestamp) for d in df["date"])


def test_parse_skips_only_malformed_date_row() -> None:
    # A digit-only but invalid date (day 99) must skip just that row, not the
    # whole file — the good rows around it must survive.
    text = (
        "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
        "20240115|AAPL|100|0|200|N\n"
        "20240199|MSFT|50|0|100|N\n"  # invalid day 99
        "20240116|NVDA|300|0|600|N\n"
    )
    df = DarkPoolDataProvider._parse(text)
    assert set(df["symbol"]) == {"AAPL", "NVDA"}


def test_coerce_schema_fills_missing_columns_with_rows() -> None:
    # Rows present but several schema columns missing must not raise on int cast.
    partial = pd.DataFrame({"symbol": ["AAPL"], "short_volume": [100]})
    df = DarkPoolDataProvider._coerce_schema(partial)
    assert list(df.columns) == list(_DARK_POOL_SCHEMA.keys())
    assert df["total_volume"].iloc[0] == 0
    assert df["short_volume"].iloc[0] == 100


def test_empty_frame_has_full_schema() -> None:
    df = DarkPoolDataProvider._empty_frame()
    assert df.empty
    assert list(df.columns) == list(_DARK_POOL_SCHEMA.keys())


def test_get_short_volume_empty_range_returns_schema() -> None:
    provider = DarkPoolDataProvider()
    # A Saturday->Sunday range has no business days -> empty, no network call.
    df = provider.get_short_volume("AAPL", "2024-01-13", "2024-01-14")
    assert df.empty
    assert list(df.columns) == list(_DARK_POOL_SCHEMA.keys())


def test_get_short_volume_filters_symbol(monkeypatch) -> None:
    provider = DarkPoolDataProvider()
    parsed = DarkPoolDataProvider._parse(_SAMPLE)

    # Avoid the network: every business day returns the same parsed sample.
    monkeypatch.setattr(provider, "_fetch_day", lambda day: parsed)

    df = provider.get_short_volume("AAPL", "2024-01-15", "2024-01-16")
    assert not df.empty
    assert set(df["symbol"]) == {"AAPL"}
    # Schema + dtypes are coerced.
    assert df["short_volume"].dtype == "int64"
    assert df["short_volume_ratio"].dtype == "float64"


def test_get_short_volume_missing_symbol_returns_empty(monkeypatch) -> None:
    provider = DarkPoolDataProvider()
    parsed = DarkPoolDataProvider._parse(_SAMPLE)
    monkeypatch.setattr(provider, "_fetch_day", lambda day: parsed)

    df = provider.get_short_volume("NVDA", "2024-01-15", "2024-01-16")
    assert df.empty
    assert list(df.columns) == list(_DARK_POOL_SCHEMA.keys())
