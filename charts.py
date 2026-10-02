"""Plotly figure builders for the training-metrics dashboard.

Every chart is a grid of small multiples, one panel per protein with its own
y-axis (losses differ several-fold between proteins), so strategies are
compared within a protein rather than across a crowded shared plot. Training
strategy is the only thing color encodes: a fixed categorical hue per strategy,
paired with a line dash so identity survives grayscale and color-vision
deficiency. One legend entry per strategy toggles it in every panel.

Every trace carries its protein in `meta`, and clickable traces have invisible
markers (Plotly only reports clicks on traces with markers), so the app can
map a click back to a protein with `clicked_protein`.
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

STRATEGY_ORDER = ["device_constrained", "mixed", "dgx_spark", "updated_metl"]
STRATEGY_COLOR = {
    "device_constrained": "#2a78d6",
    "mixed": "#eb6834",
    "dgx_spark": "#1baf7a",
    "updated_metl": "#eda100",
}
STRATEGY_DASH = {
    "device_constrained": "solid",
    "mixed": "dash",
    "dgx_spark": "dot",
    "updated_metl": "dashdot",
}
METRIC_LABELS = {"test_loss": "Test loss", "pearson_total_score": "Pearson total score"}
FACET_COLS = 4
FACET_ROW_HEIGHT = 260
FOCUSED_HEIGHT = 640
_CLICK_TARGET = dict(
    mode="lines+markers",
    marker=dict(size=8, opacity=0),
    selected=dict(marker=dict(opacity=0)),
    unselected=dict(marker=dict(opacity=0)),
)


def _ordered_strategies(strategies: list[str]) -> list[str]:
    """Selected strategies in fixed legend/color order."""
    unknown = sorted(set(strategies) - set(STRATEGY_ORDER))
    if unknown:
        raise ValueError(
            f"No chart style for training strategies {unknown}; add to STRATEGY_*."
        )
    return [s for s in STRATEGY_ORDER if s in strategies]


def clicked_protein(fig: go.Figure, points: list[dict]) -> str | None:
    """Protein of the first clicked point, from the trace's `meta`."""
    if not points:
        return None
    return fig.data[points[0]["curve_number"]].meta


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def _facet_figure(
    proteins: list[str], title: str, x_title: str, y_title: str
) -> go.Figure:
    """Empty grid with one panel per protein, shared x-axis, independent y-axes."""
    n_cols = min(FACET_COLS, len(proteins))
    n_rows = math.ceil(len(proteins) / n_cols)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=proteins,
        horizontal_spacing=0.06,
        vertical_spacing=0.32 / n_rows,
    )
    fig.update_xaxes(matches="x", showticklabels=True)
    for col in range(1, n_cols + 1):
        fig.update_xaxes(title_text=x_title, row=n_rows, col=col)
    fig.update_yaxes(title_text=y_title, col=1)
    fig.update_layout(
        title=title,
        height=FOCUSED_HEIGHT
        if len(proteins) == 1
        else FACET_ROW_HEIGHT * n_rows + 120,
        legend_title="Training strategy",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
        hovermode="closest",
    )
    return fig


def _facet_position(proteins: list[str], protein: str) -> dict[str, int]:
    n_cols = min(FACET_COLS, len(proteins))
    index = proteins.index(protein)
    return {"row": index // n_cols + 1, "col": index % n_cols + 1}


def _filter(
    df: pd.DataFrame, metric: str, proteins: list[str], strategies: list[str]
) -> pd.DataFrame:
    return df[
        df["protein"].isin(proteins)
        & df["training_strategy"].isin(strategies)
        & df[metric].notna()
    ]


def _add_run_lines(
    fig: go.Figure,
    filtered: pd.DataFrame,
    proteins: list[str],
    strategies: list[str],
    spec: dict,
) -> None:
    """Add one thin line per run, keyed by (experiment, run_id).

    The 8-character run_ids are generated independently per source and can
    collide across experiments, so run_id alone would splice two unrelated
    runs' checkpoints into a single line.
    """
    shown_in_legend = set()
    for protein in proteins:
        for strategy in strategies:
            group = filtered[
                (filtered["protein"] == protein)
                & (filtered["training_strategy"] == strategy)
            ]
            for _, run_df in group.groupby(["experiment", "run_id"]):
                run_df = run_df.sort_values(spec["sort"])
                fig.add_trace(
                    go.Scatter(
                        x=run_df[spec["x"]],
                        y=spec["y"](run_df),
                        **_CLICK_TARGET,
                        meta=protein,
                        line_shape=spec.get("line_shape", "linear"),
                        line=dict(
                            color=STRATEGY_COLOR[strategy],
                            dash=STRATEGY_DASH[strategy],
                            width=1,
                        ),
                        opacity=0.5,
                        legendgroup=strategy,
                        name=strategy,
                        showlegend=strategy not in shown_in_legend,
                        customdata=run_df[spec["customdata"]],
                        hovertemplate=spec["hovertemplate"],
                    ),
                    **_facet_position(proteins, protein),
                )
                shown_in_legend.add(strategy)


def spaghetti_fig(
    df: pd.DataFrame, metric: str, proteins: list[str], strategies: list[str]
) -> go.Figure:
    """One line per run, a panel per protein, colored and dashed by strategy."""
    strategies = _ordered_strategies(strategies)
    filtered = _filter(df, metric, proteins, strategies)
    fig = _facet_figure(
        proteins,
        f"{METRIC_LABELS[metric]} per run, by epoch",
        "Epoch",
        METRIC_LABELS[metric],
    )
    _add_run_lines(
        fig,
        filtered,
        proteins,
        strategies,
        {
            "x": "epoch",
            "sort": "epoch",
            "y": lambda run_df: run_df[metric],
            "customdata": [
                "run_id",
                "protein",
                "training_strategy",
                "experiment",
                "hostname",
                "gpu_model",
                "duration_s",
                "test_metrics_logged",
            ],
            "hovertemplate": (
                "run_id=%{customdata[0]}<br>protein=%{customdata[1]}<br>"
                "strategy=%{customdata[2]}<br>experiment=%{customdata[3]}<br>"
                "epoch=%{x}<br>" + metric + "=%{y:.4f}<br>"
                "host=%{customdata[4]}<br>gpu=%{customdata[5]}<br>"
                "duration_s=%{customdata[6]:.0f}<br>"
                "test metrics logged this epoch=%{customdata[7]}<extra></extra>"
            ),
        },
    )
    return fig


def epochs_over_time_fig(
    df: pd.DataFrame, proteins: list[str], strategies: list[str]
) -> go.Figure:
    """Cumulative epochs completed over time: one step line per run, a panel per protein.

    Each run's line uses line_shape="hv" and stops at its last checkpoint,
    so a completed run holds its final value only up to that point rather
    than extending a flat line across the rest of the time axis.
    """
    strategies = _ordered_strategies(strategies)
    filtered = df[
        df["protein"].isin(proteins)
        & df["training_strategy"].isin(strategies)
        & df["produced_at_ts"].notna()
    ]
    fig = _facet_figure(
        proteins, "Cumulative epochs completed over time", "Time", "Epochs completed"
    )
    _add_run_lines(
        fig,
        filtered,
        proteins,
        strategies,
        {
            "x": "produced_at_ts",
            "sort": "produced_at_ts",
            "y": lambda run_df: (run_df["epoch"] + 1).cummax(),
            "line_shape": "hv",
            "customdata": ["run_id", "protein", "training_strategy", "experiment"],
            "hovertemplate": (
                "run_id=%{customdata[0]}<br>protein=%{customdata[1]}<br>"
                "strategy=%{customdata[2]}<br>experiment=%{customdata[3]}<br>"
                "time=%{x}<br>epochs complete=%{y}<extra></extra>"
            ),
        },
    )
    return fig


def _epoch_stats(
    df: pd.DataFrame, metric: str, proteins: list[str], strategies: list[str]
) -> pd.DataFrame:
    filtered = _filter(df, metric, proteins, strategies)
    stats = (
        filtered.groupby(["protein", "training_strategy", "epoch"])[metric]
        .agg(mean="mean", std="std", n="count", min="min", max="max")
        .reset_index()
        .sort_values("epoch")
    )
    mean_abs = stats["mean"].abs()
    stats["cv"] = np.where(mean_abs > 1e-9, stats["std"] / mean_abs, np.nan)
    return stats


def _strategy_groups(stats: pd.DataFrame, proteins: list[str], strategies: list[str]):
    """Yield (protein, strategy, group, show_in_legend) in fixed facet/legend order."""
    shown_in_legend = set()
    for protein in proteins:
        for strategy in strategies:
            group = stats[
                (stats["protein"] == protein) & (stats["training_strategy"] == strategy)
            ]
            if group.empty:
                continue
            yield protein, strategy, group, strategy not in shown_in_legend
            shown_in_legend.add(strategy)


def trend_fig(
    df: pd.DataFrame, metric: str, proteins: list[str], strategies: list[str]
) -> go.Figure:
    """Mean per epoch with a ±1 std band, per strategy, a panel per protein."""
    strategies = _ordered_strategies(strategies)
    stats = _epoch_stats(df, metric, proteins, strategies)
    fig = _facet_figure(
        proteins,
        f"{METRIC_LABELS[metric]} — mean ± 1 std across runs",
        "Epoch",
        METRIC_LABELS[metric],
    )
    for protein, strategy, group, show in _strategy_groups(stats, proteins, strategies):
        position = _facet_position(proteins, protein)
        color = STRATEGY_COLOR[strategy]
        band_x = pd.concat([group["epoch"], group["epoch"][::-1]])
        # std is NaN where only one run reached the epoch; a zero-width band there
        # keeps the polygon outline unbroken.
        std = group["std"].fillna(0)
        band_y = pd.concat([group["mean"] + std, (group["mean"] - std)[::-1]])
        fig.add_trace(
            go.Scatter(
                x=band_x,
                y=band_y,
                mode="lines",
                fill="toself",
                fillcolor=_rgba(color, 0.15),
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=strategy,
                meta=protein,
            ),
            **position,
        )
        fig.add_trace(
            go.Scatter(
                x=group["epoch"],
                y=group["mean"],
                **_CLICK_TARGET,
                meta=protein,
                line=dict(color=color, dash=STRATEGY_DASH[strategy], width=2),
                name=strategy,
                legendgroup=strategy,
                showlegend=show,
                customdata=group[["std", "n", "min", "max"]],
                hovertemplate=(
                    "epoch=%{x}<br>mean=%{y:.4f}<br>std=%{customdata[0]:.4f}<br>"
                    "n=%{customdata[1]}<br>min=%{customdata[2]:.4f}<br>"
                    "max=%{customdata[3]:.4f}<extra>"
                    + protein
                    + " / "
                    + strategy
                    + "</extra>"
                ),
            ),
            **position,
        )
    return fig


def variance_fig(
    df: pd.DataFrame, metric: str, proteins: list[str], strategies: list[str]
) -> go.Figure:
    """Coefficient of variation (std / |mean|) per epoch, per strategy, a panel per protein."""
    strategies = _ordered_strategies(strategies)
    stats = _epoch_stats(df, metric, proteins, strategies)
    fig = _facet_figure(
        proteins,
        f"{METRIC_LABELS[metric]} — run-to-run coefficient of variation",
        "Epoch",
        "CV (std / |mean|)",
    )
    for protein, strategy, group, show in _strategy_groups(stats, proteins, strategies):
        fig.add_trace(
            go.Scatter(
                x=group["epoch"],
                y=group["cv"],
                **_CLICK_TARGET,
                meta=protein,
                line=dict(
                    color=STRATEGY_COLOR[strategy],
                    dash=STRATEGY_DASH[strategy],
                    width=2,
                ),
                name=strategy,
                legendgroup=strategy,
                showlegend=show,
                customdata=group[["n"]],
                hovertemplate=(
                    "epoch=%{x}<br>CV=%{y:.3f}<br>n=%{customdata[0]}"
                    "<extra>" + protein + " / " + strategy + "</extra>"
                ),
            ),
            **_facet_position(proteins, protein),
        )
    return fig
