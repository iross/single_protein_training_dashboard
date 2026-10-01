"""Streamlit dashboard: per-protein test_loss / pearson_total_score by epoch."""

import streamlit as st

import charts
from data import get_dashboard_data

st.set_page_config(page_title="Training metrics dashboard", layout="wide")

df, n_unmapped = get_dashboard_data()

st.sidebar.header("Filters")

all_proteins = sorted(df["protein"].unique())
selected_proteins = st.sidebar.multiselect("Proteins", all_proteins, default=all_proteins)

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
        st.plotly_chart(
            charts.spaghetti_fig(df, metric, selected_proteins, selected_strategies),
            width="stretch",
        )

        st.subheader("Trend across training")
        st.plotly_chart(
            charts.trend_fig(df, metric, selected_proteins, selected_strategies),
            width="stretch",
        )

        st.subheader("Variance across runs")
        st.plotly_chart(
            charts.variance_fig(df, metric, selected_proteins, selected_strategies),
            width="stretch",
        )

with tabs[2]:
    st.subheader("Cumulative epochs completed over time")
    st.plotly_chart(
        charts.epochs_over_time_fig(df, selected_proteins, selected_strategies),
        width="stretch",
    )
