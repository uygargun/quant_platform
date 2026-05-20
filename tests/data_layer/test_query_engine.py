"""Tests for QueryEngine input validation (SQL injection prevention)."""

import pytest

from data.query.engine import _validate_identifier, _validate_interval


class TestValidateIdentifier:
    def test_valid_alphanumeric(self):
        assert _validate_identifier("EURUSD", "symbol") == "EURUSD"
        assert _validate_identifier("histdata", "source") == "histdata"
        assert _validate_identifier("1m", "timeframe") == "1m"

    def test_valid_with_underscores(self):
        assert _validate_identifier("hist_data_v2", "source") == "hist_data_v2"

    def test_rejects_sql_injection(self):
        with pytest.raises(ValueError, match="Invalid symbol"):
            _validate_identifier("EURUSD'; DROP TABLE--", "symbol")

    def test_rejects_path_traversal(self):
        with pytest.raises(ValueError, match="Invalid source"):
            _validate_identifier("../../../etc/passwd", "source")

    def test_rejects_spaces(self):
        with pytest.raises(ValueError, match="Invalid"):
            _validate_identifier("EUR USD", "symbol")

    def test_rejects_semicolons(self):
        with pytest.raises(ValueError, match="Invalid"):
            _validate_identifier("test;evil", "source")

    def test_rejects_quotes(self):
        with pytest.raises(ValueError, match="Invalid"):
            _validate_identifier("test'quote", "source")

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="Invalid"):
            _validate_identifier("", "source")


class TestValidateInterval:
    def test_valid_intervals(self):
        assert _validate_interval("1 hour") == "1 hour"
        assert _validate_interval("30 minutes") == "30 minutes"
        assert _validate_interval("1 day") == "1 day"
        assert _validate_interval("5 second") == "5 second"

    def test_strips_whitespace(self):
        assert _validate_interval("  1 hour  ") == "1 hour"

    def test_rejects_sql_injection(self):
        with pytest.raises(ValueError, match="Invalid interval"):
            _validate_interval("1 hour'; DROP TABLE runs--")

    def test_rejects_arbitrary_text(self):
        with pytest.raises(ValueError, match="Invalid interval"):
            _validate_interval("SELECT * FROM runs")

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="Invalid interval"):
            _validate_interval("")
