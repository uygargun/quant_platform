"""Tests for UI components — HTML escaping and rendering."""

import pytest

from ui.components import _safe, card_html, history_card_html, robustness_bar


class TestSafeEscape:
    def test_escapes_html_tags(self):
        assert _safe("<script>alert('xss')</script>") == "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;"

    def test_escapes_quotes(self):
        assert _safe('test"value') == "test&quot;value"

    def test_passes_safe_strings(self):
        assert _safe("hello world") == "hello world"
        assert _safe("Sharpe Ratio") == "Sharpe Ratio"

    def test_handles_numbers(self):
        assert _safe(42) == "42"
        assert _safe(3.14) == "3.14"


class TestCardHtml:
    def test_renders_basic_card(self):
        html = card_html("Test Label", "42.0", "positive")
        assert "Test Label" in html
        assert "42.0" in html
        assert "positive" in html

    def test_escapes_malicious_label(self):
        html = card_html("<img src=x onerror=alert(1)>", "value")
        assert "<img" not in html
        assert "&lt;img" in html

    def test_escapes_malicious_value(self):
        html = card_html("label", "<script>evil()</script>")
        assert "<script>" not in html


class TestHistoryCardHtml:
    def test_renders_entry(self):
        entry = {
            "type": "backtest",
            "label": "sma_cross / data.csv",
            "id": 1,
            "timestamp": "2024-01-01 12:00:00",
            "metrics": {"sharpe": 1.5, "cagr": 0.12},
        }
        html = history_card_html(entry)
        assert "backtest" in html
        assert "sma_cross" in html

    def test_escapes_malicious_label(self):
        entry = {
            "type": "backtest",
            "label": "<img src=x onerror=alert(1)>",
            "id": 1,
            "timestamp": "2024-01-01",
            "metrics": {},
        }
        html = history_card_html(entry)
        assert "<img" not in html
        assert "&lt;img" in html


class TestRobustnessBar:
    def test_high_score_green(self):
        html = robustness_bar(85.0)
        assert "#3fb950" in html

    def test_mid_score_yellow(self):
        html = robustness_bar(55.0)
        assert "#d29922" in html

    def test_low_score_red(self):
        html = robustness_bar(30.0)
        assert "#f85149" in html

    def test_caps_at_100(self):
        html = robustness_bar(150.0)
        assert "width:100%" in html
