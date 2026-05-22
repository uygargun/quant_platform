"""Report export — standalone HTML reports for backtest/optimization results."""
from __future__ import annotations

from html import escape as _esc

import pandas as pd
import plotly.io as pio


class ReportExporter:
    """Generates standalone HTML reports from backtest results."""

    def generate_html(
        self,
        title: str,
        equity_curve: pd.Series,
        metrics: dict,
        *,
        trades: pd.DataFrame | None = None,
        regimes: pd.Series | None = None,
        benchmark: pd.Series | None = None,
        extra_sections: list[tuple[str, str]] | None = None,
    ) -> str:
        """Build a self-contained HTML report.

        Args:
            title: Report title.
            equity_curve: Strategy equity curve.
            metrics: Performance metrics dict.
            trades: Optional trade log DataFrame.
            regimes: Optional regime series for equity chart.
            benchmark: Optional buy-and-hold equity for overlay.
            extra_sections: List of (heading, html_content) to append.

        Returns:
            Complete HTML string ready for download.
        """
        from ui.charts import drawdown_chart, equity_chart

        sections: list[str] = []

        # Metrics table
        sections.append(self._metrics_table(metrics))

        # Equity chart
        eq_fig = equity_chart(
            equity_curve, title="Equity Curve",
            regimes=regimes, benchmark=benchmark,
        )
        sections.append(_fig_to_html(eq_fig))

        # Drawdown chart
        dd_fig = drawdown_chart(equity_curve)
        sections.append(_fig_to_html(dd_fig))

        # Trade log
        if trades is not None and len(trades) > 0:
            sections.append(self._trade_table(trades))

        # Extra sections
        if extra_sections:
            for heading, content in extra_sections:
                sections.append(
                    f"<h2>{_esc(heading)}</h2>\n{content}"
                )

        body = "\n".join(sections)
        return _wrap_html(title, body)

    @staticmethod
    def _metrics_table(metrics: dict) -> str:
        """Render metrics as an HTML table."""
        fmt_map = {
            "total_return": ("{:+.2%}", "Total Return"),
            "cagr": ("{:+.2%}", "CAGR"),
            "sharpe": ("{:.2f}", "Sharpe"),
            "sortino": ("{:.2f}", "Sortino"),
            "max_drawdown": ("{:+.2%}", "Max Drawdown"),
            "volatility": ("{:.2%}", "Volatility"),
            "win_rate": ("{:.1%}", "Win Rate"),
            "profit_factor": ("{:.2f}", "Profit Factor"),
            "avg_trade": ("{:+.2f}", "Avg Trade"),
            "total_trades": ("{:d}", "Total Trades"),
            "alpha": ("{:+.2%}", "Alpha"),
            "beta": ("{:.2f}", "Beta"),
            "information_ratio": ("{:.2f}", "Info Ratio"),
            "tracking_error": ("{:.2%}", "Tracking Error"),
        }
        rows = []
        for key, (fmt, label) in fmt_map.items():
            if key in metrics:
                val = metrics[key]
                try:
                    formatted = fmt.format(val)
                except (ValueError, TypeError):
                    formatted = str(val)
                rows.append(f"<tr><td>{_esc(label)}</td><td>{_esc(formatted)}</td></tr>")

        if not rows:
            return ""
        return (
            "<h2>Performance Metrics</h2>\n"
            '<table class="metrics">\n'
            + "\n".join(rows)
            + "\n</table>"
        )

    @staticmethod
    def _trade_table(trades: pd.DataFrame) -> str:
        """Render trade log as an HTML table."""
        display_cols = [
            c for c in [
                "entry_time", "exit_time", "side", "avg_entry",
                "exit_price", "shares", "pnl",
            ] if c in trades.columns
        ]
        if not display_cols:
            return ""

        header = "".join(f"<th>{_esc(c)}</th>" for c in display_cols)
        rows = []
        for _, row in trades.head(200).iterrows():
            cells = []
            for c in display_cols:
                val = row[c]
                if isinstance(val, float):
                    cells.append(f"<td>{val:.4f}</td>")
                else:
                    cells.append(f"<td>{_esc(str(val))}</td>")
            rows.append("<tr>" + "".join(cells) + "</tr>")

        note = ""
        if len(trades) > 200:
            note = f"<p><em>Showing first 200 of {len(trades)} trades.</em></p>"

        return (
            f"<h2>Trade Log ({len(trades)} trades)</h2>\n"
            f"{note}"
            f'<table class="trades">\n<tr>{header}</tr>\n'
            + "\n".join(rows)
            + "\n</table>"
        )


def _fig_to_html(fig) -> str:
    """Convert a Plotly figure to an embeddable HTML fragment."""
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")


def _wrap_html(title: str, body: str) -> str:
    """Wrap body content in a complete HTML document."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>
  body {{
    background: #0d1117;
    color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
  }}
  h1 {{
    color: #58a6ff;
    border-bottom: 1px solid #30363d;
    padding-bottom: 12px;
  }}
  h2 {{
    color: #e6edf3;
    margin-top: 32px;
    font-size: 1.2rem;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
  }}
  table.metrics {{
    max-width: 500px;
  }}
  th, td {{
    text-align: left;
    padding: 8px 12px;
    border-bottom: 1px solid #21262d;
  }}
  th {{
    background: #161b22;
    color: #8b949e;
    font-weight: 600;
  }}
  tr:hover td {{
    background: #161b22;
  }}
  .footer {{
    margin-top: 40px;
    padding-top: 12px;
    border-top: 1px solid #30363d;
    color: #8b949e;
    font-size: 0.85rem;
  }}
</style>
</head>
<body>
<h1>{_esc(title)}</h1>
{body}
<div class="footer">
  Generated by Quant Research Platform
</div>
</body>
</html>"""
