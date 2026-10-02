"""Streamlit dashboard: per-protein test_loss / pearson_total_score by epoch."""

import streamlit as st

import charts
import summaries
from data import get_dashboard_data

st.set_page_config(page_title="Training metrics dashboard", layout="wide")

df, n_unmapped = get_dashboard_data()

st.sidebar.header("Filters")

all_proteins = sorted(df["protein"].unique())
selected_proteins = st.sidebar.multiselect(
    "Proteins", all_proteins, default=all_proteins
)

all_strategies = sorted(df["training_strategy"].unique())
selected_strategies = st.sidebar.multiselect(
    "Training strategy", all_strategies, default=all_strategies
)

if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()

if n_unmapped:
    st.sidebar.caption(f"{n_unmapped} checkpoint rows had no protein mapping.")

st.title("Training metrics dashboard")

if not selected_proteins or not selected_strategies:
    st.info("Select at least one protein and one training strategy in the sidebar.")
    st.stop()

FOCUS_KEY = "focused_protein"
GRID_CHART_PREFIX = "grid-chart-"


def show_all_proteins() -> None:
    """Leave focus mode; drop grid charts' stale click selections so they don't refocus."""
    st.session_state.pop(FOCUS_KEY, None)
    for key in [k for k in st.session_state if str(k).startswith(GRID_CHART_PREFIX)]:
        del st.session_state[key]


focused = st.session_state.get(FOCUS_KEY)
if focused not in selected_proteins:
    focused = None
    st.session_state.pop(FOCUS_KEY, None)

if focused:
    chart_proteins = [focused]
    st.button(f"← Back to all proteins (showing {focused})", on_click=show_all_proteins)
else:
    chart_proteins = selected_proteins
    st.caption("Click a line in any panel to enlarge that protein.")


def show_chart(fig, name: str) -> None:
    """Render a chart; in the grid view, a click focuses the clicked protein."""
    if focused:
        st.plotly_chart(fig, width="stretch")
        return
    event = st.plotly_chart(
        fig,
        width="stretch",
        key=GRID_CHART_PREFIX + name,
        on_select="rerun",
        selection_mode="points",
    )
    protein = charts.clicked_protein(fig, event.selection.points)
    if protein:
        st.session_state[FOCUS_KEY] = protein
        st.rerun()


st.caption(
    "Test metrics are logged only when validation loss reaches a new best, so each "
    "epoch shows the test metrics of the run's best checkpoint so far."
)


def filter_run_table(run_table):
    """Filter controls in one row above the per-run table; returns the matching rows."""
    protein_col, experiment_col, status_col, search_col = st.columns([3, 3, 2, 2])
    proteins = sorted(run_table["protein"].unique())
    chosen_proteins = protein_col.multiselect(
        "Protein", proteins, default=proteins, key="runs-protein"
    )
    experiments = sorted(run_table["experiment"].unique())
    chosen_experiments = experiment_col.multiselect(
        "Experiment", experiments, default=experiments, key="runs-experiment"
    )
    status = status_col.selectbox(
        "Status", ["All", "Completed", "Incomplete"], key="runs-status"
    )
    run_id_query = search_col.text_input("run_id contains", key="runs-run-id").strip()
    mask = run_table["protein"].isin(chosen_proteins)
    mask &= run_table["experiment"].isin(chosen_experiments)
    if status != "All":
        mask &= run_table["completed"] == (status == "Completed")
    if run_id_query:
        mask &= run_table["run_id"].str.contains(run_id_query, case=False, regex=False)
    return run_table[mask]


tabs = st.tabs(
    [
        charts.METRIC_LABELS["test_loss"],
        charts.METRIC_LABELS["pearson_total_score"],
        "Training progress",
        "Run summary",
    ]
)

for tab, metric in zip(tabs[:2], ["test_loss", "pearson_total_score"]):
    with tab:
        st.subheader("Per-run trajectories")
        show_chart(
            charts.spaghetti_fig(df, metric, chart_proteins, selected_strategies),
            f"spaghetti-{metric}",
        )

        st.subheader("Mean ± std across runs")
        show_chart(
            charts.trend_fig(df, metric, chart_proteins, selected_strategies),
            f"trend-{metric}",
        )

        st.subheader("Run-to-run variation")
        show_chart(
            charts.variance_fig(df, metric, chart_proteins, selected_strategies),
            f"variance-{metric}",
        )

with tabs[2]:
    st.subheader("Cumulative epochs completed over time")
    show_chart(
        charts.epochs_over_time_fig(df, chart_proteins, selected_strategies),
        "epochs-over-time",
    )

with tabs[3]:
    filtered = df[
        df["protein"].isin(chart_proteins)
        & df["training_strategy"].isin(selected_strategies)
    ]
    run_table = summaries.run_summary(filtered)
    test_cols = {"test_loss": "Test loss", "pearson_total_score": "Pearson"}

    st.subheader("Final metrics by protein and strategy")
    st.caption(
        f"Completed runs only (reached epoch {summaries.FINAL_EPOCH}); test metrics of each "
        "run's best-validation checkpoint."
    )
    overall = summaries.overall_metrics_table(run_table)
    st.dataframe(
        overall,
        hide_index=True,
        column_config={
            "completed_runs": "Completed runs",
            **{
                f"{m}_{stat}": st.column_config.NumberColumn(
                    f"{label} {stat}",
                    format="%.5f" if m == "pearson_total_score" else "%.3f",
                )
                for m, label in test_cols.items()
                for stat in ("mean", "std")
            },
        },
    )

    st.subheader("Runs by training strategy")
    st.caption(
        "Wall-clock and compute medians are over completed runs; compute sums epoch "
        f"durations of at least {summaries.MIN_DURATION_S} s."
    )
    st.dataframe(
        summaries.strategy_summary_table(run_table),
        hide_index=True,
        column_config={
            "median_wall_h_completed": st.column_config.NumberColumn(
                "median wall clock (h)", format="%.0f"
            ),
            "median_compute_h_completed": st.column_config.NumberColumn(
                "median compute (h)", format="%.1f"
            ),
            "median_compute_fraction_completed": st.column_config.NumberColumn(
                "median compute fraction", format="%.2f"
            ),
        },
    )

    st.subheader("All runs")
    shown_runs = filter_run_table(run_table)
    st.caption(f"{len(shown_runs)} of {len(run_table)} runs")
    st.dataframe(
        shown_runs.reset_index(drop=True),
        hide_index=True,
        column_config={
            "first_ts": st.column_config.DatetimeColumn("first checkpoint"),
            "last_ts": st.column_config.DatetimeColumn("last checkpoint"),
            "best_val_loss": st.column_config.NumberColumn(format="%.3f"),
            "test_loss": st.column_config.NumberColumn(
                "test loss (best ckpt)", format="%.3f"
            ),
            "pearson_total_score": st.column_config.NumberColumn(
                "Pearson (best ckpt)", format="%.5f"
            ),
            "wall_h": st.column_config.NumberColumn("wall clock (h)", format="%.0f"),
            "compute_h": st.column_config.NumberColumn("compute (h)", format="%.1f"),
            "compute_fraction": st.column_config.NumberColumn(format="%.2f"),
        },
    )
