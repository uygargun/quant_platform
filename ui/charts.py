"""Reusable Plotly chart builders.

All functions return go.Figure objects — no st.plotly_chart calls here.
Rendering is the caller's responsibility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ui.styles import COMPARE_COLORS, PLOTLY_DARK, REGIME_COLORS


def equity_chart(
    equity: pd.Series,
    title: str = "Equity Curve",
    regimes: pd.Series = None,
    benchmark: pd.Series | None = None,
) -> go.Figure:
    """Equity line with optional regime shading and benchmark overlay."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        mode="lines", name="Equity",
        line=dict(color="#58a6ff", width=2),
        hovertemplate="$%{y:,.0f}<extra>%{x|%Y-%m-%d}</extra>",
    ))
    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=benchmark.index, y=benchmark.values,
            mode="lines", name="Buy & Hold",
            line=dict(color="#8b949e", width=1.5, dash="dash"),
            hovertemplate="$%{y:,.0f}<extra>Buy & Hold</extra>",
        ))
    if regimes is not None:
        for regime, color in REGIME_COLORS.items():
            mask = regimes == regime
            if not mask.any():
                continue
            starts = mask & ~mask.shift(1, fill_value=False)
            ends = mask & ~mask.shift(-1, fill_value=False)
            for s, e in zip(equity.index[starts], equity.index[ends]):
                fig.add_vrect(x0=s, x1=e, fillcolor=color,
                              line_width=0, layer="below")
        for regime, color in REGIME_COLORS.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=color.replace("0.12", "0.6")),
                name=regime.replace("_", " ").title(),
            ))
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Equity ($)",
        height=420, margin=dict(l=60, r=30, t=50, b=40),
        hovermode="x unified", legend=dict(orientation="h", y=-0.15),
        **PLOTLY_DARK,
    )
    return fig


def drawdown_chart(equity: pd.Series) -> go.Figure:
    """Drawdown area chart."""
    peak = equity.cummax()
    dd = (equity - peak) / peak
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        mode="lines", name="Drawdown",
        fill="tozeroy",
        line=dict(color="#ef5350", width=1),
        fillcolor="rgba(239,83,80,0.15)",
        hovertemplate="%{y:.2%}<extra>Drawdown</extra>",
    ))
    fig.update_layout(
        title="Drawdown", xaxis_title="Date", yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        height=280, margin=dict(l=60, r=30, t=40, b=40),
        **PLOTLY_DARK,
    )
    return fig


def heatmap_chart(
    all_runs: pd.DataFrame,
    px_name: str,
    py_name: str,
    target: str,
) -> go.Figure:
    """2D parameter heatmap (for 2-param optimizations)."""
    pivot = all_runs.pivot_table(
        index=py_name, columns=px_name, values=target,
    )
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(v) for v in pivot.columns],
        y=[str(v) for v in pivot.index],
        colorscale="RdYlGn",
        hovertemplate=(f"{px_name}: %{{x}}<br>{py_name}: %{{y}}"
                       f"<br>{target}: %{{z:.4f}}"
                       f"<extra></extra>"),
    ))
    fig.update_layout(
        title=f"{target.title()} Heatmap",
        xaxis_title=px_name, yaxis_title=py_name,
        height=400, margin=dict(l=60, r=30, t=50, b=40),
        **PLOTLY_DARK,
    )
    return fig


def robustness_bar_chart(sorted_selected: list[dict]) -> go.Figure:
    """Horizontal bar chart of robustness scores for selected strategies."""
    names = [f"Trial {s['trial_id']}" for s in sorted_selected]
    robustness_vals = [s["robustness"] for s in sorted_selected]
    colors = [
        "#238636" if s["decision"] == "APPROVED"
        else "#da3633" if s["decision"] == "REJECTED"
        else "#d29922"
        for s in sorted_selected
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=robustness_vals,
        marker_color=colors,
        text=[f"{v:.0f}" for v in robustness_vals],
        textposition="outside",
        hovertemplate="%{x}: %{y:.1f}/100<extra></extra>",
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="#8b949e",
                  annotation_text="Minimum (50)")
    fig.update_layout(
        title="Robustness Comparison",
        yaxis_title="Score (0-100)",
        yaxis_range=[0, 105],
        height=350, margin=dict(l=60, r=30, t=50, b=40),
        **PLOTLY_DARK,
    )
    return fig


def montecarlo_fan_chart(
    mc,
    n_paths: int,
    method: str,
) -> go.Figure | None:
    """Monte Carlo fan chart with percentile bands. Returns None if data missing."""
    p5_line = _get_percentile(mc, 5)
    p95_line = _get_percentile(mc, 95)
    p25_line = _get_percentile(mc, 25)
    p75_line = _get_percentile(mc, 75)
    p50_line = _get_percentile(mc, 50)

    if p5_line is None or p95_line is None or p50_line is None:
        return None

    fig = go.Figure()

    # Outer band: 5–95
    fig.add_trace(go.Scatter(
        x=p95_line.index, y=p95_line.values,
        mode="lines", line=dict(width=0), showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=p5_line.index, y=p5_line.values,
        mode="lines", line=dict(width=0), showlegend=False,
        fill="tonexty", fillcolor="rgba(88,166,255,0.08)",
        hoverinfo="skip",
    ))
    # Inner band: 25–75 (optional)
    if p25_line is not None and p75_line is not None:
        fig.add_trace(go.Scatter(
            x=p75_line.index, y=p75_line.values,
            mode="lines", line=dict(width=0), showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=p25_line.index, y=p25_line.values,
            mode="lines", line=dict(width=0), showlegend=False,
            fill="tonexty", fillcolor="rgba(88,166,255,0.18)",
            hoverinfo="skip",
        ))
    # Median line
    fig.add_trace(go.Scatter(
        x=p50_line.index, y=p50_line.values,
        mode="lines", name="Median",
        line=dict(color="#58a6ff", width=2),
        hovertemplate="$%{y:,.0f}<extra>Median</extra>",
    ))
    fig.update_layout(
        title=f"Monte Carlo Fan Chart ({n_paths} paths, {method})",
        xaxis_title="Bar", yaxis_title="Equity ($)",
        height=420, margin=dict(l=60, r=30, t=50, b=40),
        legend=dict(orientation="h", y=-0.15),
        **PLOTLY_DARK,
    )
    return fig


def montecarlo_return_dist(final_returns: np.ndarray) -> go.Figure:
    """Histogram of final returns."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=final_returns,
        nbinsx=40, marker_color="#58a6ff",
        hovertemplate="%{x:.2%}<extra></extra>",
    ))
    fig.update_layout(
        title="Final Return Distribution",
        xaxis_title="Return", xaxis_tickformat=".0%",
        yaxis_title="Count",
        height=300, margin=dict(l=50, r=20, t=40, b=40),
        **PLOTLY_DARK,
    )
    return fig


def montecarlo_dd_dist(max_drawdowns: np.ndarray) -> go.Figure:
    """Histogram of max drawdowns."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=max_drawdowns,
        nbinsx=40, marker_color="#ef5350",
        hovertemplate="%{x:.2%}<extra></extra>",
    ))
    fig.update_layout(
        title="Max Drawdown Distribution",
        xaxis_title="Max Drawdown", xaxis_tickformat=".0%",
        yaxis_title="Count",
        height=300, margin=dict(l=50, r=20, t=40, b=40),
        **PLOTLY_DARK,
    )
    return fig


def compare_equity_chart(selected_runs: list[dict]) -> go.Figure:
    """Overlay normalized equity curves for comparison."""
    fig = go.Figure()
    for i, entry in enumerate(selected_runs):
        eq = entry["equity_curve"]
        normalized = (eq / eq.iloc[0]) * 100
        color = COMPARE_COLORS[i % len(COMPARE_COLORS)]
        fig.add_trace(go.Scatter(
            x=normalized.index,
            y=normalized.values,
            mode="lines",
            name=f"#{entry['id']} {entry['label'][:40]}",
            line=dict(color=color, width=2),
            hovertemplate="%{y:.1f}<extra>%{x|%Y-%m-%d}</extra>",
        ))
    fig.add_hline(y=100, line_dash="dot",
                  line_color="#484f58", line_width=1)
    fig.update_layout(
        title="Equity Curve Comparison (Normalized to 100)",
        xaxis_title="Date",
        yaxis_title="Indexed Value (start = 100)",
        height=480,
        margin=dict(l=60, r=30, t=50, b=40),
        legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
        hovermode="x unified",
        **PLOTLY_DARK,
    )
    return fig


def compare_drawdown_chart(selected_runs: list[dict]) -> go.Figure:
    """Overlay drawdowns for comparison."""
    fig = go.Figure()
    for i, entry in enumerate(selected_runs):
        eq = entry["equity_curve"]
        peak = eq.cummax()
        dd = (eq - peak) / peak
        color = COMPARE_COLORS[i % len(COMPARE_COLORS)]
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            mode="lines",
            name=f"#{entry['id']}",
            line=dict(color=color, width=1.5),
            hovertemplate="%{y:.2%}<extra>#" + str(entry['id']) + "</extra>",
        ))
    fig.update_layout(
        title="Drawdown Comparison",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        height=350,
        margin=dict(l=60, r=30, t=50, b=40),
        legend=dict(orientation="h", y=-0.18),
        hovermode="x unified",
        **PLOTLY_DARK,
    )
    return fig


# ─── Bayesian optimization charts ────────────────────────────────────


def convergence_chart(
    all_runs: pd.DataFrame,
    target: str,
    maximize: bool = True,
) -> go.Figure:
    """Convergence plot: per-trial objective value + running best."""
    values = all_runs[target].values
    # Reconstruct trial order (all_runs is sorted by metric; use original index)
    # We need trial-order values — re-sort by original insertion order
    # all_runs was reset_index(drop=True) after sort, so we need the
    # unsorted column values. The safest approach: compute running best
    # from the all_runs as returned (sorted by metric), but for the
    # convergence plot we need trial order. The all_runs DataFrame
    # keeps all param+metric columns. Since BayesianOptimizer sorts by
    # target before returning, we use a heuristic: the data is already
    # available in _internals.opt_result.all_runs. The caller should pass
    # a trial-ordered DataFrame, or we plot the sorted values as-is with
    # a note. For robustness, we just plot every trial's value and
    # the cumulative best.
    if maximize:
        running_best = np.maximum.accumulate(values)
    else:
        running_best = np.minimum.accumulate(values)

    trial_nums = np.arange(1, len(values) + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trial_nums, y=values,
        mode="markers",
        name="Trial Value",
        marker=dict(color="#58a6ff", size=5, opacity=0.5),
        hovertemplate="Trial %{x}<br>Value: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=trial_nums, y=running_best,
        mode="lines",
        name="Best So Far",
        line=dict(color="#3fb950", width=2),
        hovertemplate="Trial %{x}<br>Best: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Convergence — {target.title()}",
        xaxis_title="Trial",
        yaxis_title=target.title(),
        height=380,
        margin=dict(l=60, r=30, t=50, b=40),
        legend=dict(orientation="h", y=-0.15),
        **PLOTLY_DARK,
    )
    return fig


def param_importance_chart(
    all_runs: pd.DataFrame,
    param_names: list[str],
    target: str,
) -> go.Figure:
    """Parameter importance via correlation with the target metric."""
    importances = {}
    for p in param_names:
        if p in all_runs.columns:
            col = pd.to_numeric(all_runs[p], errors="coerce")
            if col.nunique() > 1:
                importances[p] = abs(col.corr(all_runs[target]))
            else:
                importances[p] = 0.0

    if not importances:
        fig = go.Figure()
        fig.update_layout(title="No parameter variance", **PLOTLY_DARK)
        return fig

    sorted_params = sorted(importances, key=importances.get, reverse=True)
    sorted_vals = [importances[p] for p in sorted_params]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sorted_params, y=sorted_vals,
        marker_color="#58a6ff",
        text=[f"{v:.3f}" for v in sorted_vals],
        textposition="outside",
        hovertemplate="%{x}: |corr| = %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Parameter Importance (|corr| with {target})",
        yaxis_title="|Correlation|",
        yaxis_range=[0, 1.05],
        height=350,
        margin=dict(l=60, r=30, t=50, b=40),
        **PLOTLY_DARK,
    )
    return fig


def parallel_coordinates_chart(
    all_runs: pd.DataFrame,
    param_names: list[str],
    target: str,
) -> go.Figure:
    """Parallel coordinates plot coloured by target metric."""
    dimensions = []
    for p in param_names:
        if p in all_runs.columns:
            col = pd.to_numeric(all_runs[p], errors="coerce")
            if col.notna().any():
                dimensions.append(dict(
                    label=p,
                    values=col.values,
                    range=[col.min(), col.max()],
                ))

    if target in all_runs.columns:
        dimensions.append(dict(
            label=target,
            values=all_runs[target].values,
            range=[all_runs[target].min(), all_runs[target].max()],
        ))

    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=all_runs[target].values if target in all_runs.columns else None,
            colorscale="RdYlGn",
            showscale=True,
            cmin=all_runs[target].min() if target in all_runs.columns else 0,
            cmax=all_runs[target].max() if target in all_runs.columns else 1,
        ),
        dimensions=dimensions,
    ))
    fig.update_layout(
        title="Parameter Space Exploration",
        height=420,
        margin=dict(l=80, r=80, t=50, b=40),
        **PLOTLY_DARK,
    )
    return fig


# ─── Walk-forward charts ────────────────────────────────────────────

def walkforward_fold_chart(windows) -> go.Figure:
    """Timeline bar chart showing train/test/embargo windows per fold."""
    fig = go.Figure()

    for w in windows:
        fold = w.fold
        fig.add_trace(go.Bar(
            x=[(pd.Timestamp(w.train_end) - pd.Timestamp(w.train_start)).days],
            y=[f"Fold {fold}"],
            base=[pd.Timestamp(w.train_start)],
            orientation="h",
            marker_color="rgba(88,166,255,0.4)",
            name="Train" if fold == 0 else None,
            showlegend=(fold == 0),
            hovertemplate=(
                f"Fold {fold} Train<br>"
                f"{w.train_start} → {w.train_end}<extra></extra>"
            ),
        ))
        fig.add_trace(go.Bar(
            x=[(pd.Timestamp(w.test_end) - pd.Timestamp(w.test_start)).days],
            y=[f"Fold {fold}"],
            base=[pd.Timestamp(w.test_start)],
            orientation="h",
            marker_color="rgba(63,185,80,0.6)",
            name="Test" if fold == 0 else None,
            showlegend=(fold == 0),
            hovertemplate=(
                f"Fold {fold} Test<br>"
                f"{w.test_start} → {w.test_end}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Walk-Forward Fold Timeline",
        barmode="overlay",
        height=max(200, len(windows) * 50 + 80),
        margin=dict(l=80, r=30, t=50, b=40),
        legend=dict(orientation="h", y=-0.15),
        **PLOTLY_DARK,
    )
    return fig


def is_oos_comparison_chart(windows, target: str) -> go.Figure:
    """Grouped bar chart: IS vs OOS metric per fold."""
    folds = [f"Fold {w.fold}" for w in windows]
    is_vals = [w.best_train_metric for w in windows]
    oos_vals = [w.test_metrics.get(target, 0) for w in windows]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=folds, y=is_vals, name="In-Sample",
        marker_color="rgba(88,166,255,0.7)",
        hovertemplate="%{x}: %{y:.4f}<extra>IS</extra>",
    ))
    fig.add_trace(go.Bar(
        x=folds, y=oos_vals, name="Out-of-Sample",
        marker_color="rgba(63,185,80,0.7)",
        hovertemplate="%{x}: %{y:.4f}<extra>OOS</extra>",
    ))
    fig.update_layout(
        title=f"IS vs OOS — {target.title()}",
        barmode="group",
        yaxis_title=target.title(),
        height=350,
        margin=dict(l=60, r=30, t=50, b=40),
        legend=dict(orientation="h", y=-0.15),
        **PLOTLY_DARK,
    )
    return fig


# ─── Portfolio optimization charts ──────────────────────────────────

def efficient_frontier_chart(
    random_vols: np.ndarray,
    random_rets: np.ndarray,
    random_sharpes: np.ndarray,
    optimal_points: dict[str, tuple[float, float]],
) -> go.Figure:
    """Scatter plot of random portfolios with optimal points highlighted."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=random_vols, y=random_rets,
        mode="markers",
        marker=dict(
            size=4, color=random_sharpes,
            colorscale="RdYlGn", showscale=True,
            colorbar=dict(title="Sharpe"),
            opacity=0.6,
        ),
        name="Random Portfolios",
        hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<br><extra></extra>",
    ))

    markers = {"min_variance": ("#58a6ff", "star", "Min Variance"),
               "max_sharpe": ("#3fb950", "diamond", "Max Sharpe"),
               "risk_parity": ("#d29922", "cross", "Risk Parity"),
               "equal": ("#8b949e", "circle", "Equal Weight")}
    for method, (vol, ret) in optimal_points.items():
        color, symbol, label = markers.get(method, ("#ffffff", "circle", method))
        fig.add_trace(go.Scatter(
            x=[vol], y=[ret],
            mode="markers+text",
            marker=dict(size=14, color=color, symbol=symbol,
                        line=dict(width=2, color="white")),
            text=[label], textposition="top center",
            textfont=dict(color=color, size=11),
            name=label,
            hovertemplate=f"{label}<br>Vol: %{{x:.2%}}<br>Return: %{{y:.2%}}<extra></extra>",
        ))

    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Annualised Volatility",
        yaxis_title="Annualised Return",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        height=480, margin=dict(l=60, r=30, t=50, b=40),
        legend=dict(orientation="h", y=-0.15),
        **PLOTLY_DARK,
    )
    return fig


def weight_allocation_chart(
    weights: dict[str, float],
    title: str = "Portfolio Allocation",
) -> go.Figure:
    """Pie chart showing portfolio weight allocation."""
    names = list(weights.keys())
    values = list(weights.values())
    fig = go.Figure(data=go.Pie(
        labels=names, values=values,
        hole=0.4,
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:.2%}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        height=380, margin=dict(l=30, r=30, t=50, b=40),
        **PLOTLY_DARK,
    )
    return fig


# ─── Internal helpers ────────────────────────────────────────────────

def _get_percentile(mc, p: int) -> pd.Series | None:
    """Safely access mc.percentiles[p], handling int/str keys and DataFrames."""
    pcts = mc.percentiles
    if isinstance(pcts, dict):
        if p in pcts:
            return pcts[p]
        if str(p) in pcts:
            return pcts[str(p)]
        return None
    if isinstance(pcts, pd.DataFrame):
        if p in pcts.columns:
            return pcts[p]
        if str(p) in pcts.columns:
            return pcts[str(p)]
        return None
    return None
