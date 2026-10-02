"""Streamlit dashboard: per-protein test_loss / pearson_total_score by epoch."""

import streamlit as st

import charts
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

tabs = st.tabs(
    [
        charts.METRIC_LABELS["test_loss"],
        charts.METRIC_LABELS["pearson_total_score"],
        "Training progress",
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
