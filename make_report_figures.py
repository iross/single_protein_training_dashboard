"""Render device_constrained vs mixed comparison figures for reports.

Reads data/dashboard_data.csv (regenerate with `just build-data`) and writes PNG
and PDF files to figures/report/:

    uv run python make_report_figures.py

Per-protein training curves go to <protein>.png/.pdf. Cross-protein summaries:
final_epoch_summary, hardware_sensitivity, mixed_gpu_heatmap, gpu_switch_effect,
time_overhead, and throughput_by_gpu.

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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

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
NEUTRAL = "#8a8985"
SWITCH_COLOR = "#4a3aa7"
FINAL_EPOCH = 29
MIN_EPOCHS_PER_GPU = 20
# Shorter (or negative) epoch durations are timing artifacts in the logged data.
MIN_DURATION_S = 60
RNG_SEED = 0
# GPU architecture families, oldest first, matched by substring of gpu_model.
GPU_FAMILIES = {
    "Pascal": ["1080 Ti", "P100"],
    "Turing": ["2080 Ti", "Quadro RTX"],
    "Ampere": ["A100", "A30", "A40", "A5000"],
    "Ada": ["L40"],
    "Hopper": ["H100", "H200"],
    "Blackwell": ["Blackwell"],
}
FAMILY_COLORS = {
    "Pascal": "#2a78d6",
    "Turing": "#eb6834",
    "Ampere": "#1baf7a",
    "Ada": "#eda100",
    "Hopper": "#e87ba4",
    "Blackwell": "#008300",
    "Other": "#c3c2b7",
}


def load_comparison_data() -> pd.DataFrame:
    """Load dashboard rows for the device_constrained vs mixed comparison."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run `just build-data` first.")
    df = pd.read_csv(DATA_PATH, dtype={"run_id": str}, parse_dates=["produced_at_ts"])
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


def style_axes(ax: plt.Axes, title: str, xlabel: str = "Epoch") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=11)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
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
        loc="upper left",
        bbox_to_anchor=(0.01, 0.93),
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_PRIMARY,
    )
    fig.suptitle(
        f"{protein}: mean ± 1 std across runs (test metrics of best checkpoint so far)",
        x=0.01,
        ha="left",
        color=TEXT_PRIMARY,
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return fig


def suptitle(fig: plt.Figure, text: str) -> None:
    fig.suptitle(
        text, x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=13, fontweight="bold"
    )


def strategy_legend(fig: plt.Figure, df: pd.DataFrame) -> None:
    """Figure-level legend with one marker per strategy and its completed-run count."""
    handles = [
        Patch(color=spec["color"], label=f"{spec['label']} (n={n} runs)")
        for strategy, spec in STRATEGIES.items()
        if (n := df.loc[df["training_strategy"] == strategy, "run_id"].nunique())
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.955),
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_PRIMARY,
    )


def run_key(df: pd.DataFrame) -> pd.Series:
    """Unique run identity; run_ids can collide across experiments."""
    return df["experiment"] + "/" + df["run_id"]


def final_epoch_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per completed run, at FINAL_EPOCH.

    Test metrics there are the run's best-validation checkpoint's (see
    build_dashboard_data.carry_forward_test_metrics).
    """
    return df[df["epoch"] == FINAL_EPOCH].assign(run=run_key)


def bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    """95% percentile bootstrap CI of the mean."""
    if len(values) < 2:
        return float("nan"), float("nan")
    means = rng.choice(values, size=(5000, len(values))).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def gpu_family(gpu_model: str) -> str:
    for family, needles in GPU_FAMILIES.items():
        if any(needle in gpu_model for needle in needles):
            return family
    return "Other"


def short_gpu_name(gpu_model: str) -> str:
    return (
        gpu_model.replace("NVIDIA ", "")
        .replace("GeForce ", "")
        .replace(" Server Edition", "")
    )


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {OUT_DIR / name}.png/.pdf")


def final_epoch_summary_figure(df: pd.DataFrame) -> plt.Figure:
    """Final-epoch metric per completed run, by protein and strategy, with mean and CI."""
    final = final_epoch_rows(df)
    proteins = sorted(final["protein"].unique())
    rng = np.random.default_rng(RNG_SEED)
    fig, axes = plt.subplots(
        len(METRICS),
        len(proteins),
        figsize=(2.1 * len(proteins), 6.4),
        facecolor=SURFACE,
    )
    for row, metric in enumerate(METRICS):
        for col, protein in enumerate(proteins):
            ax = axes[row, col]
            for x, (strategy, spec) in enumerate(STRATEGIES.items()):
                values = final.loc[
                    (final["protein"] == protein)
                    & (final["training_strategy"] == strategy),
                    metric,
                ].to_numpy()
                if not len(values):
                    ax.text(
                        x,
                        0.5,
                        "no completed\nruns",
                        transform=ax.get_xaxis_transform(),
                        ha="center",
                        fontsize=8,
                        color=TEXT_SECONDARY,
                    )
                    continue
                jitter = rng.uniform(-0.12, 0.12, len(values))
                ax.scatter(
                    x + jitter, values, s=18, color=spec["color"], alpha=0.6, lw=0
                )
                low, high = bootstrap_mean_ci(values, rng)
                ax.errorbar(
                    x + 0.3,
                    values.mean(),
                    yerr=[[values.mean() - low], [high - values.mean()]],
                    fmt="o",
                    color=spec["color"],
                    ms=5,
                    capsize=3,
                    lw=1.5,
                )
            ax.set_xticks([0, 1], ["DC", "Mixed"])
            ax.set_xlim(-0.5, 1.6)
            style_axes(ax, protein if row == 0 else "", xlabel="")
        axes[row, 0].set_ylabel(METRICS[metric], color=TEXT_SECONDARY)
    strategy_legend(fig, final)
    suptitle(
        fig,
        "Test metrics of each completed run's best-validation checkpoint "
        "(mean, 95% bootstrap CI)",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def hardware_sensitivity_figure(df: pd.DataFrame) -> plt.Figure:
    """Device-constrained final metric per GPU model, as % deviation from the mixed mean."""
    final = final_epoch_rows(df)
    mixed_mean = (
        final[final["training_strategy"] == "mixed"]
        .groupby("protein")[list(METRICS)]
        .mean()
    )
    deviation = final.copy()
    for metric in METRICS:
        baseline = deviation["protein"].map(mixed_mean[metric])
        deviation[metric] = (deviation[metric] - baseline) / baseline * 100
    dc = deviation[deviation["training_strategy"] == "device_constrained"]
    mixed = deviation[deviation["training_strategy"] == "mixed"]
    order = dc.groupby("gpu_model")["duration_s"].median().sort_values().index.tolist()
    rng = np.random.default_rng(RNG_SEED)
    fig, axes = plt.subplots(
        len(METRICS), 1, figsize=(11, 7), sharex=True, facecolor=SURFACE
    )
    for ax, metric in zip(axes, METRICS):
        low, high = np.percentile(mixed[metric].dropna(), [5, 95])
        ax.axhspan(
            low, high, color=NEUTRAL, alpha=0.18, lw=0, label="Mixed runs, 5th–95th pct"
        )
        ax.axhline(0, color=NEUTRAL, lw=1)
        x = dc["gpu_model"].map({gpu: i for i, gpu in enumerate(order)})
        ax.scatter(
            x + rng.uniform(-0.15, 0.15, len(dc)),
            dc[metric],
            s=22,
            color=STRATEGIES["device_constrained"]["color"],
            alpha=0.75,
            lw=0,
            label="Device-constrained run (one protein each)",
        )
        style_axes(
            ax, f"{METRICS[metric]}: % deviation from the protein's mixed-run mean", ""
        )
        ax.set_ylabel("% deviation", color=TEXT_SECONDARY)
    axes[-1].set_xticks(
        range(len(order)), [short_gpu_name(g) for g in order], rotation=30, ha="right"
    )
    axes[-1].set_xlabel(
        "GPU model (ordered by median epoch time, fastest first)", color=TEXT_SECONDARY
    )
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    suptitle(fig, "Does the GPU model alone shift final results?")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def mixed_gpu_heatmap_figure(df: pd.DataFrame) -> plt.Figure:
    """GPU architecture family per epoch for each mixed run, grouped by protein."""
    mixed = df[df["training_strategy"] == "mixed"].assign(run=run_key)
    mixed = mixed.assign(family=mixed["gpu_model"].map(gpu_family))
    families = list(FAMILY_COLORS)
    grid = mixed.pivot_table(
        index=["protein", "run"], columns="epoch", values="family", aggfunc="first"
    )
    codes = grid.apply(lambda col: col.map({f: i for i, f in enumerate(families)}))
    fig, ax = plt.subplots(figsize=(10, 0.13 * len(grid) + 2), facecolor=SURFACE)
    cmap = matplotlib.colors.ListedColormap(list(FAMILY_COLORS.values()))
    cmap.set_bad(SURFACE)
    ax.imshow(
        np.ma.masked_invalid(codes.to_numpy(dtype=float)),
        aspect="auto",
        cmap=cmap,
        vmin=-0.5,
        vmax=len(families) - 0.5,
        interpolation="nearest",
    )
    proteins = grid.index.get_level_values("protein")
    boundaries = [i for i in range(1, len(proteins)) if proteins[i] != proteins[i - 1]]
    for b in boundaries:
        ax.axhline(b - 0.5, color=SURFACE, lw=2.5)
    starts = [0, *boundaries]
    ends = [*boundaries, len(proteins)]
    ax.set_yticks(
        [(s + e - 1) / 2 for s, e in zip(starts, ends)], [proteins[s] for s in starts]
    )
    style_axes(ax, "", xlabel="Epoch")
    ax.grid(False)
    used = [f for f in families if f in set(mixed["family"])]
    ax.legend(
        handles=[Patch(color=FAMILY_COLORS[f], label=f) for f in used],
        ncol=len(used),
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        frameon=False,
        fontsize=9,
    )
    switches = (
        grid.ne(grid.shift(axis=1)) & grid.notna() & grid.shift(axis=1).notna()
    ).sum(axis=1)
    suptitle(
        fig,
        f"GPU architecture per epoch, mixed runs (median {switches.median():.0f} "
        "family switches per run)",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def gpu_switch_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Mixed runs' epoch-to-epoch val-loss change, relative to device-constrained runs.

    Uses val_loss because it is logged every epoch; test_loss is only logged on
    new-best epochs.

    Loss falls fastest early in training, so each epoch's change is compared with the
    median change of device-constrained runs of the same protein at the same epoch,
    as a percentage of the previous epoch's loss.
    """
    ordered = df.assign(run=run_key).sort_values(["run", "epoch"])
    previous = ordered.groupby("run")[["val_loss", "gpu_model", "epoch"]].shift()
    ordered = ordered.assign(
        delta_pct=(ordered["val_loss"] - previous["val_loss"])
        / previous["val_loss"]
        * 100,
        switched=ordered["gpu_model"] != previous["gpu_model"],
        consecutive=ordered["epoch"] - previous["epoch"] == 1,
    )
    ordered = ordered[ordered["consecutive"] & ordered["delta_pct"].notna()]
    baseline = (
        ordered[ordered["training_strategy"] == "device_constrained"]
        .groupby(["protein", "epoch"])["delta_pct"]
        .median()
        .rename("baseline")
    )
    mixed = ordered[ordered["training_strategy"] == "mixed"].join(
        baseline, on=["protein", "epoch"]
    )
    return mixed.assign(excess_pct=mixed["delta_pct"] - mixed["baseline"]).dropna(
        subset=["excess_pct"]
    )


def gpu_switch_effect_figure(df: pd.DataFrame) -> plt.Figure:
    """Excess val-loss change at epochs after a GPU-model switch vs same-GPU epochs."""
    deltas = gpu_switch_deltas(df)
    groups = [*sorted(deltas["protein"].unique()), "All proteins"]
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=SURFACE)
    tick_labels = []
    for i, group in enumerate(groups):
        subset = (
            deltas if group == "All proteins" else deltas[deltas["protein"] == group]
        )
        for offset, switched, color in [
            (-0.18, False, NEUTRAL),
            (0.18, True, SWITCH_COLOR),
        ]:
            values = subset.loc[subset["switched"] == switched, "excess_pct"]
            if values.empty:
                continue
            ax.boxplot(
                values,
                positions=[i + offset],
                widths=0.3,
                showfliers=False,
                patch_artist=True,
                medianprops=dict(color=TEXT_PRIMARY),
                boxprops=dict(facecolor=color, alpha=0.5, edgecolor=color),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
            )
        counts = subset["switched"].value_counts()
        tick_labels.append(f"{group}\nn={counts.get(False, 0)} / {counts.get(True, 0)}")
    ax.axhline(0, color=NEUTRAL, lw=1)
    ax.set_xticks(range(len(groups)), tick_labels)
    style_axes(
        ax,
        "Validation-loss change vs device-constrained median at the same epoch "
        "(n = same-GPU / changed epochs)",
        xlabel="",
    )
    ax.set_ylabel("Excess change (% of previous loss)", color=TEXT_SECONDARY)
    ax.legend(
        handles=[
            Patch(color=NEUTRAL, alpha=0.5, label="Same GPU model as previous epoch"),
            Patch(color=SWITCH_COLOR, alpha=0.5, label="GPU model changed"),
        ],
        frameon=False,
        fontsize=9,
        loc="upper right",
    )
    suptitle(fig, "Does switching GPU model disturb training? (mixed runs)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def run_timing(df: pd.DataFrame) -> pd.DataFrame:
    """Wall-clock and compute hours for each run that completed training."""
    runs = df.assign(run=run_key)
    completed = runs.groupby("run")["epoch"].transform("max") == FINAL_EPOCH
    runs = runs[completed].sort_values("produced_at_ts")
    runs = runs.assign(
        valid_duration_s=runs["duration_s"].where(
            runs["duration_s"] >= MIN_DURATION_S, 0
        )
    )
    timing = runs.groupby("run").agg(
        strategy=("training_strategy", "first"),
        first_ts=("produced_at_ts", "first"),
        last_ts=("produced_at_ts", "last"),
        first_duration_s=("duration_s", "first"),
        compute_s=("valid_duration_s", "sum"),
    )
    first_duration = timing["first_duration_s"].clip(lower=0)
    started = timing["first_ts"] - pd.to_timedelta(first_duration, unit="s")
    timing["wall_h"] = (timing["last_ts"] - started).dt.total_seconds() / 3600
    timing["compute_h"] = timing["compute_s"] / 3600
    timing["compute_fraction"] = timing["compute_h"] / timing["wall_h"]
    return timing


def time_overhead_figure(df: pd.DataFrame) -> plt.Figure:
    """Wall-clock vs compute hours per completed run, and the fraction spent computing."""
    timing = run_timing(df)
    panels = [
        ("wall_h", "Wall clock to finish (h)"),
        ("compute_h", f"Compute, sum of epochs ≥{MIN_DURATION_S} s (h)"),
        ("compute_fraction", "Fraction of wall clock computing"),
    ]
    rng = np.random.default_rng(RNG_SEED)
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 4.2), facecolor=SURFACE)
    for ax, (column, title) in zip(axes, panels):
        for x, (strategy, spec) in enumerate(STRATEGIES.items()):
            values = timing.loc[timing["strategy"] == strategy, column]
            ax.boxplot(
                values,
                positions=[x],
                widths=0.5,
                showfliers=False,
                medianprops=dict(color=TEXT_PRIMARY),
                boxprops=dict(color=spec["color"]),
                whiskerprops=dict(color=spec["color"]),
                capprops=dict(color=spec["color"]),
            )
            ax.scatter(
                x + rng.uniform(-0.15, 0.15, len(values)),
                values,
                s=12,
                color=spec["color"],
                alpha=0.5,
                lw=0,
            )
            ax.text(
                x + 0.27,
                values.median(),
                f" {values.median():.2f}"
                if column == "compute_fraction"
                else f" {values.median():.0f}",
                va="center",
                ha="left",
                fontsize=8,
                color=TEXT_PRIMARY,
            )
        ax.set_xticks([0, 1], ["DC", "Mixed"])
        style_axes(ax, title, xlabel="")
    strategy_legend(fig, df[df.assign(run=run_key)["run"].isin(timing.index)])
    suptitle(
        fig,
        f"Time to complete {FINAL_EPOCH + 1} epochs (completed runs, medians labeled)",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def throughput_by_gpu_figure(df: pd.DataFrame) -> plt.Figure:
    """Epoch duration by GPU model, relative to each protein's median epoch duration."""
    epochs = df[(df["gpu_model"] != "none") & (df["duration_s"] >= MIN_DURATION_S)]
    relative = epochs["duration_s"] / epochs.groupby("protein")["duration_s"].transform(
        "median"
    )
    epochs = epochs.assign(relative=relative)
    counts = epochs["gpu_model"].value_counts()
    kept = counts[counts >= MIN_EPOCHS_PER_GPU].index
    dropped = sorted(set(counts.index) - set(kept))
    medians = (
        epochs[epochs["gpu_model"].isin(kept)].groupby("gpu_model")["relative"].median()
    )
    order = medians.sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(order) + 1.8), facecolor=SURFACE)
    for i, gpu in enumerate(order):
        values = epochs.loc[epochs["gpu_model"] == gpu, "relative"]
        color = FAMILY_COLORS[gpu_family(gpu)]
        ax.boxplot(
            values,
            positions=[i],
            widths=0.6,
            vert=False,
            whis=(5, 95),
            showfliers=False,
            patch_artist=True,
            medianprops=dict(color=TEXT_PRIMARY),
            boxprops=dict(facecolor=color, alpha=0.6, edgecolor=color),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
        )
    ax.set_yticks(
        range(len(order)), [f"{short_gpu_name(g)} (n={counts[g]})" for g in order]
    )
    ax.invert_yaxis()
    ax.set_xscale("log")
    ticks = [0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.axvline(1, color=NEUTRAL, lw=1)
    style_axes(
        ax,
        "Epoch duration ÷ the protein's median epoch duration (log scale; 1 = typical)",
        xlabel="Relative epoch duration",
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    families = [f for f in FAMILY_COLORS if f in {gpu_family(g) for g in order}]
    ax.legend(
        handles=[Patch(color=FAMILY_COLORS[f], alpha=0.6, label=f) for f in families],
        frameon=False,
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
    )
    note = (
        f"Whiskers: 5th–95th percentile. Excluded: GPU models with fewer than "
        f"{MIN_EPOCHS_PER_GPU} epochs ({', '.join(map(short_gpu_name, dropped))}), "
        f"gpu_model 'none', and epochs shorter than {MIN_DURATION_S} s (timing artifacts).\n"
        "All runs used PyTorch built against CUDA 11.6, which has no sm_90 kernels, so "
        "Hopper GPUs (H100, H200) did not run native kernels and are slower than their hardware allows."
    )
    fig.text(0.01, 0.01, note, fontsize=7, color=TEXT_SECONDARY)
    suptitle(fig, "Epoch throughput by GPU model (fastest at top)")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return fig


SUMMARY_FIGURES = {
    "final_epoch_summary": final_epoch_summary_figure,
    "hardware_sensitivity": hardware_sensitivity_figure,
    "mixed_gpu_heatmap": mixed_gpu_heatmap_figure,
    "gpu_switch_effect": gpu_switch_effect_figure,
    "time_overhead": time_overhead_figure,
    "throughput_by_gpu": throughput_by_gpu_figure,
}


def main() -> None:
    df = load_comparison_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for protein in sorted(df["protein"].unique()):
        save(protein_figure(df[df["protein"] == protein], protein), protein)
    for name, build in SUMMARY_FIGURES.items():
        save(build(df), name)


if __name__ == "__main__":
    main()
