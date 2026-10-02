"""Per-run and per-strategy summaries of the dashboard data, shared by the
dashboard (app.py) and the report (make_report_figures.py). Pure pandas.
"""

import pandas as pd

TEST_METRICS = ["test_loss", "pearson_total_score"]
FINAL_EPOCH = 29
# Shorter (or negative) epoch durations are timing artifacts in the logged data.
MIN_DURATION_S = 60


def run_key(df: pd.DataFrame) -> pd.Series:
    """Unique run identity; run_ids can collide across experiments."""
    return df["experiment"] + "/" + df["run_id"]


def run_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per run: hardware used, timing, and final (best-checkpoint) metrics.

    Compute hours sum epoch durations of at least MIN_DURATION_S; wall clock runs
    from the start of the first epoch to the last checkpoint.
    """
    runs = df.assign(run=run_key).sort_values(["run", "produced_at_ts"])
    runs = runs.assign(
        valid_duration_s=runs["duration_s"].where(
            runs["duration_s"] >= MIN_DURATION_S, 0
        ),
        gpu_switch=runs.groupby("run")["gpu_model"]
        .shift()
        .pipe(lambda prev: prev.notna() & (prev != runs["gpu_model"])),
    )
    summary = runs.groupby("run").agg(
        experiment=("experiment", "first"),
        run_id=("run_id", "first"),
        protein=("protein", "first"),
        strategy=("training_strategy", "first"),
        epochs=("epoch", "nunique"),
        last_epoch=("epoch", "max"),
        gpu_models=("gpu_model", "nunique"),
        hosts=("hostname", "nunique"),
        gpu_switches=("gpu_switch", "sum"),
        first_ts=("produced_at_ts", "first"),
        last_ts=("produced_at_ts", "last"),
        first_duration_s=("duration_s", "first"),
        compute_s=("valid_duration_s", "sum"),
        best_val_loss=("val_loss", "min"),
    )
    last_rows = runs.loc[runs.groupby("run")["epoch"].idxmax()].set_index("run")
    summary[TEST_METRICS] = last_rows[TEST_METRICS]
    summary["completed"] = summary["last_epoch"] == FINAL_EPOCH
    first_duration = summary["first_duration_s"].clip(lower=0)
    started = summary["first_ts"] - pd.to_timedelta(first_duration, unit="s")
    summary["wall_h"] = (summary["last_ts"] - started).dt.total_seconds() / 3600
    summary["compute_h"] = summary["compute_s"] / 3600
    summary["compute_fraction"] = summary["compute_h"] / summary["wall_h"]
    return summary.drop(columns=["first_duration_s", "compute_s"])


def overall_metrics_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Per protein and strategy: completed runs and final metric mean/std."""
    completed = summary[summary["completed"]]
    grouped = completed.groupby(["protein", "strategy"])
    table = grouped.size().rename("completed_runs").to_frame()
    for metric in TEST_METRICS:
        table[f"{metric}_mean"] = grouped[metric].mean()
        table[f"{metric}_std"] = grouped[metric].std()
    return table.reset_index()


def strategy_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Per strategy: run counts, completion, hardware heterogeneity, and timing medians."""
    grouped = summary.groupby("strategy")
    completed = summary[summary["completed"]].groupby("strategy")
    return pd.DataFrame(
        {
            "runs": grouped.size(),
            "completed_runs": completed.size().reindex(
                grouped.size().index, fill_value=0
            ),
            "median_epochs": grouped["epochs"].median(),
            "median_gpu_models": grouped["gpu_models"].median(),
            "median_hosts": grouped["hosts"].median(),
            "median_gpu_switches": grouped["gpu_switches"].median(),
            "median_wall_h_completed": completed["wall_h"].median(),
            "median_compute_h_completed": completed["compute_h"].median(),
            "median_compute_fraction_completed": completed["compute_fraction"].median(),
        }
    ).reset_index()
