"""Variance analysis: stable (baseline) vs mobile runs, with hardware transition overlay."""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from metrics import EVAL_DIR, list_metrics, load_all_runs
from load_hardware import (
    TRANSITION_CATEGORIES,
    categorized_transition_counts,
    classify_runs,
    load_epoch_hardware,
)

CAT_STYLE = {
    "both":      {"color": "#d62728", "label": "site + GPU changed"},
    "site_only": {"color": "#ff7f0e", "label": "site only"},
    "gpu_only":  {"color": "#9467bd", "label": "GPU only"},
    "same":      {"color": "#aec7e8", "label": "same site + GPU"},
}


def epoch_arrays(runs: dict[str, dict[int, float]]) -> tuple[list[int], np.ndarray, np.ndarray]:
    all_epochs = sorted(set(e for v in runs.values() for e in v))
    per_epoch = [[v[e] for v in runs.values() if e in v] for e in all_epochs]
    means = np.array([np.mean(v) for v in per_epoch])
    stds = np.array([np.std(v) for v in per_epoch])
    return all_epochs, means, stds


def plot_ribbon(ax: plt.Axes, runs: dict, color: str, label: str) -> None:
    epochs, means, stds = epoch_arrays(runs)
    ax.fill_between(epochs, means - stds, means + stds, alpha=0.2, color=color)
    ax.plot(epochs, means, linewidth=2, color=color, label=label)
    for values in runs.values():
        ep = sorted(values)
        ax.plot(ep, [values[e] for e in ep], linewidth=0.5, alpha=0.3, color=color)


def plot_cv(ax: plt.Axes, runs: dict, color: str, label: str) -> None:
    epochs, means, stds = epoch_arrays(runs)
    cvs = stds / np.abs(means)
    ax.plot(epochs, cvs, linewidth=1.5, color=color, label=label)


def add_transition_overlay(ax: plt.Axes, cat_counts: dict, key: str) -> None:
    non_same = {e: sum(v for c, v in cats.items() if c != "same")
                for e, cats in cat_counts.items()}
    max_count = max(non_same.values()) if non_same else 1
    for epoch, count in non_same.items():
        if count == 0:
            continue
        alpha = 0.15 + 0.55 * (count / max_count)
        ax.axvline(epoch, color="red", linewidth=0.8, alpha=alpha, zorder=1)


def plot_transition_bars(ax: plt.Axes, cat_counts: dict, title: str, n_mobile: int) -> None:
    t_epochs = sorted(cat_counts)
    bottoms = np.zeros(len(t_epochs))
    for cat in TRANSITION_CATEGORIES:
        if cat == "same":
            continue  # excluded — stable baseline is a separate group now
        heights = np.array([cat_counts[e].get(cat, 0) for e in t_epochs], dtype=float)
        ax.bar(t_epochs, heights, bottom=bottoms, width=0.7,
               color=CAT_STYLE[cat]["color"], label=CAT_STYLE[cat]["label"], alpha=0.85)
        bottoms += heights
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_ylabel("# mobile runs")
    ax.set_title(f"{title} transitions (mobile runs only, n={n_mobile})")
    ax.legend(fontsize=7, loc="upper right")


def make_plot(
    metric: str,
    stable_runs: dict,
    mobile_runs: dict,
    cat_counts: dict,
    title_suffix: str,
    suffix: str,
) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 1]},
    )

    if stable_runs:
        plot_ribbon(ax1, stable_runs, color="#1f77b4", label=f"stable (n={len(stable_runs)})")
    if mobile_runs:
        plot_ribbon(ax1, mobile_runs, color="#d62728", label=f"mobile (n={len(mobile_runs)})")
    add_transition_overlay(ax1, cat_counts, suffix)
    ax1.set_ylabel(metric)
    ax1.set_title(f"{metric} — mean ± std  |  stable vs mobile runs")
    ax1.legend(fontsize=8)

    if stable_runs:
        plot_cv(ax2, stable_runs, color="#1f77b4", label="stable")
    if mobile_runs:
        plot_cv(ax2, mobile_runs, color="#d62728", label="mobile")
    add_transition_overlay(ax2, cat_counts, suffix)
    ax2.set_ylabel("CV (std / |mean|)")
    ax2.set_title("Coefficient of Variation")
    ax2.legend(fontsize=8)

    plot_transition_bars(ax3, cat_counts, title_suffix, len(mobile_runs))
    ax3.set_xlabel("Epoch")

    fig.tight_layout()
    out = EVAL_DIR / f"{metric.replace('/', '_')}_variance_{suffix}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Variance analysis of a metric across runs.")
    parser.add_argument("metric", nargs="?", default="test_loss", help="Metric name to analyze")
    parser.add_argument("--list", action="store_true", help="List available metrics and exit")
    args = parser.parse_args()

    if args.list:
        metrics = list_metrics()
        print("\n".join(metrics) if metrics else "No metric files found.")
        return

    all_runs = load_all_runs(args.metric)
    if not all_runs:
        print(f"No data found for metric '{args.metric}'. Use --list to see available metrics.")
        return

    epoch_hardware = load_epoch_hardware()
    stable_uuids, mobile_uuids = classify_runs(epoch_hardware)

    # Runs with no hardware data get lumped into stable (conservative)
    no_hw = set(all_runs) - set(epoch_hardware)
    stable_uuids |= no_hw

    stable_runs = {u: v for u, v in all_runs.items() if u in stable_uuids}
    mobile_runs = {u: v for u, v in all_runs.items() if u in mobile_uuids}

    print(f"Stable (baseline): {len(stable_runs)} runs  |  Mobile: {len(mobile_runs)} runs")
    if no_hw:
        print(f"  (no hardware data, counted as stable: {', '.join(sorted(no_hw))})")

    mobile_hw = {u: v for u, v in epoch_hardware.items() if u in mobile_uuids}
    cat_counts = categorized_transition_counts(mobile_hw)

    site_counts = {
        e: {**cats, "gpu_only": 0, "same": cats["same"] + cats["gpu_only"]}
        for e, cats in cat_counts.items()
    }
    gpu_counts = {
        e: {**cats, "site_only": 0, "same": cats["same"] + cats["site_only"]}
        for e, cats in cat_counts.items()
    }

    make_plot(args.metric, stable_runs, mobile_runs, site_counts, "Site", "by_site")
    make_plot(args.metric, stable_runs, mobile_runs, gpu_counts, "GPU model", "by_gpu")


if __name__ == "__main__":
    main()
