"""Data access layer for the training-metrics dashboard.

Reads the precomputed dashboard_data.csv built by build_dashboard_data.py.
This module has no DuckDB/SQLite dependency so the app runs identically
locally and in the browser (stlite/Pyodide, which can't load DuckDB's SQLite
extension).
"""

from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).parent
DATA_DIR = REPO_ROOT / "data"


@st.cache_data(ttl=300)
def get_dashboard_data() -> tuple[pd.DataFrame, int]:
    """Read the combined per-checkpoint metrics table.

    Returns:
        Tuple of (combined dataframe, count of rows with protein="unmapped").

    Raises:
        FileNotFoundError: If dashboard_data.csv hasn't been built yet.
    """
    path = DATA_DIR / "dashboard_data.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python build_dashboard_data.py` first."
        )
    df = pd.read_csv(path, dtype={"run_id": str})
    n_unmapped = int((df["protein"] == "unmapped").sum())
    return df, n_unmapped
