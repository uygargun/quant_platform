"""Tests for the gap analysis module."""

import polars as pl
import pytest

from data.query.gap_report import detect_gaps, gap_summary


@pytest.fixture
def continuous_df() -> pl.DataFrame:
    """DataFrame with no gaps."""
    return pl.DataFrame({
        "timestamp_utc": pl.datetime_range(
            start=pl.datetime(2024, 1, 2),
            end=pl.datetime(2024, 1, 2, 0, 9),
            interval="1m",
            eager=True,
        ).dt.replace_time_zone("UTC"),
        "open": [1.1] * 10,
        "high": [1.2] * 10,
        "low": [1.0] * 10,
        "close": [1.15] * 10,
        "volume": [100.0] * 10,
    })


@pytest.fixture
def gapped_df() -> pl.DataFrame:
    """DataFrame with a 5-minute gap in the middle."""
    ts1 = pl.datetime_range(
        start=pl.datetime(2024, 1, 2, 10, 0),
        end=pl.datetime(2024, 1, 2, 10, 4),
        interval="1m", eager=True,
    ).dt.replace_time_zone("UTC")

    # 5-minute gap (10:05 through 10:09 missing)
    ts2 = pl.datetime_range(
        start=pl.datetime(2024, 1, 2, 10, 10),
        end=pl.datetime(2024, 1, 2, 10, 14),
        interval="1m", eager=True,
    ).dt.replace_time_zone("UTC")

    timestamps = ts1.extend(ts2)
    n = len(timestamps)
    return pl.DataFrame({
        "timestamp_utc": timestamps,
        "open": [1.1] * n,
        "high": [1.2] * n,
        "low": [1.0] * n,
        "close": [1.15] * n,
        "volume": [100.0] * n,
    })


@pytest.fixture
def weekend_gap_df() -> pl.DataFrame:
    """DataFrame with a weekend gap (Friday to Monday)."""
    # Friday 21:00 UTC
    ts1 = pl.datetime_range(
        start=pl.datetime(2024, 1, 5, 20, 55),
        end=pl.datetime(2024, 1, 5, 21, 0),
        interval="1m", eager=True,
    ).dt.replace_time_zone("UTC")

    # Monday 00:00 UTC (after weekend)
    ts2 = pl.datetime_range(
        start=pl.datetime(2024, 1, 8, 0, 0),
        end=pl.datetime(2024, 1, 8, 0, 5),
        interval="1m", eager=True,
    ).dt.replace_time_zone("UTC")

    timestamps = ts1.extend(ts2)
    n = len(timestamps)
    return pl.DataFrame({
        "timestamp_utc": timestamps,
        "open": [1.1] * n,
        "high": [1.2] * n,
        "low": [1.0] * n,
        "close": [1.15] * n,
        "volume": [100.0] * n,
    })


class TestDetectGaps:
    def test_no_gaps(self, continuous_df: pl.DataFrame) -> None:
        gaps = detect_gaps(continuous_df, "1m")
        assert gaps.is_empty()

    def test_detects_gap(self, gapped_df: pl.DataFrame) -> None:
        gaps = detect_gaps(gapped_df, "1m")
        assert len(gaps) == 1
        assert gaps["gap_bars"][0] == 6  # 6 minutes gap
        assert not gaps["is_weekend"][0]

    def test_weekend_classified(self, weekend_gap_df: pl.DataFrame) -> None:
        gaps = detect_gaps(weekend_gap_df, "1m")
        assert len(gaps) == 1
        assert gaps["is_weekend"][0]

    def test_empty_df(self) -> None:
        empty = pl.DataFrame(schema={
            "timestamp_utc": pl.Datetime("us", "UTC"),
            "open": pl.Float64,
        })
        gaps = detect_gaps(empty, "1m")
        assert gaps.is_empty()

    def test_single_row(self) -> None:
        single = pl.DataFrame({
            "timestamp_utc": [pl.Series([None], dtype=pl.Datetime("us", "UTC"))[0]],
            "open": [1.1],
        })
        gaps = detect_gaps(single, "1m")
        assert gaps.is_empty()


class TestGapSummary:
    def test_summary_no_gaps(self) -> None:
        empty_gaps = pl.DataFrame(schema={
            "gap_start": pl.Datetime("us", "UTC"),
            "gap_end": pl.Datetime("us", "UTC"),
            "gap_seconds": pl.Int64,
            "gap_bars": pl.Int64,
            "is_weekend": pl.Boolean,
        })
        summary = gap_summary(empty_gaps)
        assert summary["total_gaps"] == 0
        assert summary["weekend_gaps"] == 0
        assert summary["unexpected_gaps"] == 0

    def test_summary_with_gaps(self, gapped_df: pl.DataFrame) -> None:
        gaps = detect_gaps(gapped_df, "1m")
        summary = gap_summary(gaps)
        assert summary["total_gaps"] == 1
        assert summary["unexpected_gaps"] == 1
        assert summary["weekend_gaps"] == 0
