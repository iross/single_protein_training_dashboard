"""Render per-protein device_constrained vs mixed comparison figures for reports.

Reads data/dashboard_data.csv (regenerate with `just build-data`) and writes one
PNG and one PDF per protein to figures/report/:

    uv run python make_report_figures.py

Only the device_constrained and heterogeneous experiments and the
device_constrained and mixed strategies are included, so DGX Spark and
updated-METL runs stay out of the comparison. Runs are grouped
by their observed training_strategy (see
build_dashboard_data.classify_training_strategy).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 -- backend must be set before pyplot import
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).parent
DATA_PATH = REPO_ROOT / "data" / "dashboard_data.csv"
OUT_DIR = REPO_ROOT / "figures" / "report"

EXPERIMENTS = ["device_constrained", "heterogeneous"]
STRATEGIES = {
    "device_constrained": {
        "label": "Device-constrained",
        "color": "#2a78d6",
        "dash": "-",
    },
    "mixed": {"label": "Mixed hardware", "color": "#eb6834", "dash": "--"},
}
METRICS = {"test_loss": "Test loss", "pearson_total_score": "Pearson total score"}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def load_comparison_data() -> pd.DataFrame:
    """Load dashboard rows for the device_constrained vs mixed comparison."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run `just build-data` first.")
    df = pd.read_csv(DATA_PATH, dtype={"run_id": str})
    df = df[
        df["experiment"].isin(EXPERIMENTS)
        & df["training_strategy"].isin(STRATEGIES)
        & (df["protein"] != "unmapped")
    ]
    if df.empty:
        raise ValueError(f"No device_constrained/mixed rows in {DATA_PATH}.")
    return df


def epoch_stats(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean, std, and run count per (training_strategy, epoch)."""
    return (
        df.dropna(subset=[metric])
        .groupby(["training_strategy", "epoch"])[metric]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )


def style_axes(ax: plt.Axes, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=11)
    ax.set_xlabel("Epoch", color=TEXT_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def plot_metric(ax: plt.Axes, df: pd.DataFrame, metric: str) -> None:
    stats = epoch_stats(df, metric)
    for strategy, spec in STRATEGIES.items():
        group = stats[stats["training_strategy"] == strategy].sort_values("epoch")
        if group.empty:
            continue
        n_runs = df.loc[df["training_strategy"] == strategy, "run_id"].nunique()
        ax.fill_between(
            group["epoch"],
            group["mean"] - group["std"],
            group["mean"] + group["std"],
            color=spec["color"],
            alpha=0.15,
            linewidth=0,
        )
        ax.plot(
            group["epoch"],
            group["mean"],
            color=spec["color"],
            linestyle=spec["dash"],
            linewidth=2,
            label=f"{spec['label']} (n={n_runs} runs)",
        )
    style_axes(ax, METRICS[metric])


def protein_figure(df: pd.DataFrame, protein: str) -> plt.Figure:
    """Two panels (test loss, Pearson) comparing strategies for one protein."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    for ax, metric in zip(axes, METRICS):
        plot_metric(ax, df, metric)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_PRIMARY,
    )
    fig.suptitle(
        f"{protein}: mean ± 1 std across runs",
        x=0.01,
        ha="left",
        color=TEXT_PRIMARY,
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def main() -> None:
    df = load_comparison_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for protein in sorted(df["protein"].unique()):
        fig = protein_figure(df[df["protein"] == protein], protein)
        for ext in ("png", "pdf"):
            fig.savefig(OUT_DIR / f"{protein}.{ext}", dpi=200, facecolor=SURFACE)
        plt.close(fig)
        print(f"Wrote {OUT_DIR / protein}.png/.pdf")


if __name__ == "__main__":
    main()
