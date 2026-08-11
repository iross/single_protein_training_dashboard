"""Plotly figure builders for the training-metrics dashboard.

Color always encodes protein; line dash always encodes training_strategy
("solid" = device_constrained, "dash" = mixed). With only one strategy
present in the data, every line/ribbon renders solid and the dash encoding
is inert until a second experiment is added.
"""

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_PALETTE = pc.qualitative.Set2 + pc.qualitative.Set3
_KNOWN_PROTEIN_ORDER = ["avgfp", "dlg4", "gb1", "grb2", "pab1", "pten", "tem-1", "ube4b"]
STRATEGY_DASH = {"device_constrained": "solid", "mixed": "dash"}
METRIC_LABELS = {"test_loss": "Test loss", "pearson_total_score": "Pearson total score"}


def protein_color_map(proteins: list[str]) -> dict[str, str]:
    """Assign each protein a stable color, reused across all three charts."""
    ordered = [p for p in _KNOWN_PROTEIN_ORDER if p in proteins]
    ordered += sorted(p for p in proteins if p not in _KNOWN_PROTEIN_ORDER)
    return {p: _PALETTE[i % len(_PALETTE)] for i, p in enumerate(ordered)}


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
    """One line per run, colored by protein, dashed by training strategy."""
    filtered = _filter(df, metric, proteins, strategies)
    colors = protein_color_map(proteins)
    fig = go.Figure()

    for (protein, strategy), group in filtered.groupby(["protein", "training_strategy"]):
        first_run = True
        for run_id, run_df in group.groupby("run_id"):
            run_df = run_df.sort_values("epoch")
            fig.add_trace(
                go.Scatter(
                    x=run_df["epoch"],
                    y=run_df[metric],
                    mode="lines",
                    line=dict(color=colors[protein], dash=_dash_for(strategy), width=1),
                    opacity=0.35,
                    legendgroup=protein,
                    name=protein,
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
        legend_title="Protein",
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
                line=dict(color=colors[protein], dash=_dash_for(strategy), width=2.5),
                name=f"{protein} ({strategy})",
                legendgroup=protein,
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
        color = colors[protein]
        upper = group["mean"] + group["std"]
        lower = group["mean"] - group["std"]

        fig.add_trace(
            go.Scatter(
                x=group["epoch"], y=upper, mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
                legendgroup=protein,
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
                legendgroup=protein,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=group["epoch"], y=group["mean"], mode="lines",
                line=dict(color=color, dash=_dash_for(strategy), width=2.5),
                name=f"{protein} ({strategy})", legendgroup=protein,
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
                name=f"{protein} ({strategy})", legendgroup=protein, showlegend=False,
                hovertemplate="epoch=%{x}<br>CV=%{y:.3f}<extra>" + protein + " / " + strategy + "</extra>",
            ),
            row=2, col=1,
        )

    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text=METRIC_LABELS[metric], row=1, col=1)
    fig.update_yaxes(title_text="CV (std / |mean|)", row=2, col=1)
    return fig
