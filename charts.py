"""Plotly figure builders for the training-metrics dashboard.

Protein sets the base hue, reused across all three charts. Training strategy
modulates that same hue's lightness (device_constrained = full saturation,
mixed = lightened toward white) *and* line dash, so two strategies for the
same protein are clearly distinguishable by color, not just a subtle dash
difference. With only one strategy present in the data, every line/ribbon
renders at full saturation and the strategy encoding is inert until a second
experiment is added.
"""

import re

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_PALETTE = pc.qualitative.Set2 + pc.qualitative.Set3
_KNOWN_PROTEIN_ORDER = ["avgfp", "dlg4", "gb1", "grb2", "pab1", "pten", "tem-1", "ube4b"]
STRATEGY_DASH = {"device_constrained": "solid", "mixed": "dash"}
STRATEGY_LIGHTEN = {"device_constrained": 0.0, "mixed": 0.55}
METRIC_LABELS = {"test_loss": "Test loss", "pearson_total_score": "Pearson total score"}
_RGB_RE = re.compile(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)")


def protein_color_map(proteins: list[str]) -> dict[str, str]:
    """Assign each protein a stable base color, reused across all three charts."""
    ordered = [p for p in _KNOWN_PROTEIN_ORDER if p in proteins]
    ordered += sorted(p for p in proteins if p not in _KNOWN_PROTEIN_ORDER)
    return {p: _PALETTE[i % len(_PALETTE)] for i, p in enumerate(ordered)}


def _lighten(color: str, factor: float) -> str:
    """Blend an rgb(...) color toward white by `factor` (0=unchanged, 1=white)."""
    match = _RGB_RE.match(color)
    if not match or factor <= 0:
        return color
    r, g, b = (round(int(c) + (255 - int(c)) * factor) for c in match.groups())
    return f"rgb({r},{g},{b})"


def _color_for(protein_base_color: str, strategy: str) -> str:
    """Protein sets the hue; strategy lightens it so experiments stay distinct."""
    return _lighten(protein_base_color, STRATEGY_LIGHTEN.get(strategy, 0.3))


def _dash_for(strategy: str) -> str:
    return STRATEGY_DASH.get(strategy, "dot")


def _filter(df: pd.DataFrame, metric: str, proteins: list[str], strategies: list[str]) -> pd.DataFrame:
    return df[
        df["protein"].isin(proteins)
        & df["training_strategy"].isin(strategies)
        & df[metric].notna()
    ]


def spaghetti_fig(
    df: pd.DataFrame, metric: str, proteins: list[str], strategies: list[str]
) -> go.Figure:
    """One line per run, colored by protein, dashed by training strategy.

    Runs are keyed by (experiment, run_id) rather than run_id alone -- the
    8-character run_ids are generated independently per source and can
    collide across experiments, which would otherwise splice two unrelated
    runs' checkpoints into a single line.
    """
    filtered = _filter(df, metric, proteins, strategies)
    colors = protein_color_map(proteins)
    fig = go.Figure()

    for (protein, strategy), group in filtered.groupby(["protein", "training_strategy"]):
        first_run = True
        for (experiment, run_id), run_df in group.groupby(["experiment", "run_id"]):
            run_df = run_df.sort_values("epoch")
            fig.add_trace(
                go.Scatter(
                    x=run_df["epoch"],
                    y=run_df[metric],
                    mode="lines",
                    line=dict(
                        color=_color_for(colors[protein], strategy),
                        dash=_dash_for(strategy),
                        width=1,
                    ),
                    opacity=0.35,
                    legendgroup=f"{protein}-{strategy}",
                    name=f"{protein} ({strategy})",
                    showlegend=first_run,
                    customdata=run_df[["run_id", "protein", "training_strategy", "experiment",
                                       "hostname", "gpu_model", "duration_s"]],
                    hovertemplate=(
                        "run_id=%{customdata[0]}<br>protein=%{customdata[1]}<br>"
                        "strategy=%{customdata[2]}<br>experiment=%{customdata[3]}<br>"
                        "epoch=%{x}<br>" + metric + "=%{y:.4f}<br>"
                        "host=%{customdata[4]}<br>gpu=%{customdata[5]}<br>"
                        "duration_s=%{customdata[6]:.0f}<extra></extra>"
                    ),
                )
            )
            first_run = False

    fig.update_layout(
        title=f"{METRIC_LABELS[metric]} per run, by epoch",
        xaxis_title="Epoch",
        yaxis_title=METRIC_LABELS[metric],
        legend_title="Protein / strategy",
    )
    return fig


def epochs_over_time_fig(df: pd.DataFrame, proteins: list[str], strategies: list[str]) -> go.Figure:
    """Cumulative epochs completed over time: one step line per run.

    Runs are keyed by (experiment, run_id) rather than run_id alone -- the
    8-character run_ids are generated independently per source and collide
    across experiments (and occasionally within the same protein/strategy),
    so run_id alone would silently splice two unrelated runs into one line.

    Each run's line uses line_shape="hv" and stops at its last checkpoint,
    so a completed run holds its final value only up to that point rather
    than extending a flat line across the rest of the time axis.
    """
    filtered = df[
        df["protein"].isin(proteins)
        & df["training_strategy"].isin(strategies)
        & df["produced_at_ts"].notna()
    ].copy()
    filtered["epochs_complete"] = filtered["epoch"] + 1
    colors = protein_color_map(proteins)
    fig = go.Figure()

    for (protein, strategy), group in filtered.groupby(["protein", "training_strategy"]):
        first_run = True
        for (experiment, run_id), run_df in group.groupby(["experiment", "run_id"]):
            run_df = run_df.sort_values("produced_at_ts")
            cumulative = run_df["epochs_complete"].cummax()
            fig.add_trace(
                go.Scatter(
                    x=run_df["produced_at_ts"],
                    y=cumulative,
                    mode="lines",
                    line_shape="hv",
                    line=dict(
                        color=_color_for(colors[protein], strategy),
                        dash=_dash_for(strategy),
                        width=1,
                    ),
                    opacity=0.35,
                    legendgroup=f"{protein}-{strategy}",
                    name=f"{protein} ({strategy})",
                    showlegend=first_run,
                    customdata=run_df[["run_id", "protein", "training_strategy", "experiment"]],
                    hovertemplate=(
                        "run_id=%{customdata[0]}<br>protein=%{customdata[1]}<br>"
                        "strategy=%{customdata[2]}<br>experiment=%{customdata[3]}<br>"
                        "time=%{x}<br>epochs complete=%{y}<extra></extra>"
                    ),
                )
            )
            first_run = False

    fig.update_layout(
        title="Cumulative epochs completed over time",
        xaxis_title="Time",
        yaxis_title="Epochs completed",
        legend_title="Protein / strategy",
    )
    return fig


def _epoch_stats(df: pd.DataFrame, metric: str, proteins: list[str], strategies: list[str]) -> pd.DataFrame:
    filtered = _filter(df, metric, proteins, strategies)
    stats = (
        filtered.groupby(["protein", "training_strategy", "epoch"])[metric]
        .agg(mean="mean", std="std", n="count", min="min", max="max")
        .reset_index()
    )
    return stats


def trend_fig(
    df: pd.DataFrame, metric: str, proteins: list[str], strategies: list[str]
) -> go.Figure:
    """Bold mean-per-epoch line per (protein, training_strategy) group."""
    stats = _epoch_stats(df, metric, proteins, strategies)
    colors = protein_color_map(proteins)
    fig = go.Figure()

    for (protein, strategy), group in stats.groupby(["protein", "training_strategy"]):
        group = group.sort_values("epoch")
        fig.add_trace(
            go.Scatter(
                x=group["epoch"],
                y=group["mean"],
                mode="lines+markers",
                line=dict(
                    color=_color_for(colors[protein], strategy),
                    dash=_dash_for(strategy),
                    width=2.5,
                ),
                name=f"{protein} ({strategy})",
                legendgroup=f"{protein}-{strategy}",
                customdata=group[["std", "n", "min", "max"]],
                hovertemplate=(
                    "epoch=%{x}<br>mean=%{y:.4f}<br>std=%{customdata[0]:.4f}<br>"
                    "n=%{customdata[1]}<br>min=%{customdata[2]:.4f}<br>"
                    "max=%{customdata[3]:.4f}<extra>" + protein + " / " + strategy + "</extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"{METRIC_LABELS[metric]} — mean across runs, by epoch",
        xaxis_title="Epoch",
        yaxis_title=f"Mean {METRIC_LABELS[metric]}",
        legend_title="Protein / strategy",
    )
    return fig


def variance_fig(
    df: pd.DataFrame, metric: str, proteins: list[str], strategies: list[str]
) -> go.Figure:
    """Mean±std ribbon (top) and coefficient of variation (bottom), shared x-axis."""
    stats = _epoch_stats(df, metric, proteins, strategies)
    colors = protein_color_map(proteins)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
        subplot_titles=(f"{METRIC_LABELS[metric]} — mean ± std", "Coefficient of variation"),
    )

    for (protein, strategy), group in stats.groupby(["protein", "training_strategy"]):
        group = group.sort_values("epoch")
        color = _color_for(colors[protein], strategy)
        legendgroup = f"{protein}-{strategy}"
        upper = group["mean"] + group["std"]
        lower = group["mean"] - group["std"]

        fig.add_trace(
            go.Scatter(
                x=group["epoch"], y=upper, mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
                legendgroup=legendgroup,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=group["epoch"], y=lower, mode="lines",
                line=dict(width=0), fill="tonexty",
                fillcolor=color.replace("rgb", "rgba").replace(")", ",0.2)")
                if color.startswith("rgb") else color,
                opacity=0.2, showlegend=False, hoverinfo="skip",
                legendgroup=legendgroup,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=group["epoch"], y=group["mean"], mode="lines",
                line=dict(color=color, dash=_dash_for(strategy), width=2.5),
                name=f"{protein} ({strategy})", legendgroup=legendgroup,
                customdata=group[["std", "n"]],
                hovertemplate=(
                    "epoch=%{x}<br>mean=%{y:.4f}<br>std=%{customdata[0]:.4f}<br>"
                    "n=%{customdata[1]}<extra>" + protein + " / " + strategy + "</extra>"
                ),
            ),
            row=1, col=1,
        )

        eps = 1e-9
        cv = np.where(group["mean"].abs() > eps, group["std"] / group["mean"].abs(), np.nan)
        cv = pd.Series(cv).replace([np.inf, -np.inf], np.nan)
        fig.add_trace(
            go.Scatter(
                x=group["epoch"], y=cv, mode="lines",
                line=dict(color=color, dash=_dash_for(strategy), width=2),
                name=f"{protein} ({strategy})", legendgroup=legendgroup, showlegend=False,
                hovertemplate="epoch=%{x}<br>CV=%{y:.3f}<extra>" + protein + " / " + strategy + "</extra>",
            ),
            row=2, col=1,
        )

    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text=METRIC_LABELS[metric], row=1, col=1)
    fig.update_yaxes(title_text="CV (std / |mean|)", row=2, col=1)
    return fig
