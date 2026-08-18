"""Data access layer for the training-metrics dashboard.

This module is the hotswap boundary: adding an experiment or pointing at a
different database means editing SOURCES here, not touching charts.py or app.py.
"""

import re
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).parent
DATA_DIR = REPO_ROOT / "data"

SOURCES = [
    {
        "db_path": DATA_DIR / "provenance.db",
        "dag_path": DATA_DIR / "many_protein_pretraining_with_ospool_device_constrained_runs.dag",
        "experiment": "device_constrained",
    },
    {
        "db_path": DATA_DIR / "mixed.db",
        "dag_path": DATA_DIR / "many_protein_pretraining_with_ospool.dag",
        "experiment": "heterogeneous",
    },
    # future: {"db_path": ..., "dag_path": ..., "experiment": "mixed"},
]

METRICS_QUERY = """
    -- Some (run_id, epoch) pairs have multiple checkpoint rows: a canonical
    -- end-of-epoch checkpoint plus periodic checkpoints/time_checkpoints/
    -- snapshots taken mid-epoch and stamped with the same epoch number. Their
    -- metric values are identical, so QUALIFY keeps one row per (run_id, epoch)
    -- -- the canonical checkpoint if one exists, else the latest snapshot --
    -- rather than letting duplicates inflate per-epoch counts/std.
    SELECT
        run_id,
        epoch,
        TRY(json_extract_string(training_json, '$.test_loss')::DOUBLE) AS test_loss,
        TRY(json_extract_string(training_json, '$."test/pearson_total_score"')::DOUBLE)
            AS pearson_total_score,
        hostname,
        gpu_model,
        duration_s,
        val_loss,
        train_loss_epoch,
        checkpoint_path
    FROM checkpoints
    WHERE training_json IS NOT NULL
    QUALIFY row_number() OVER (
        PARTITION BY run_id, epoch
        ORDER BY checkpoint_path NOT LIKE '%time_checkpoints%' DESC, produced_at_ts DESC
    ) = 1
    ORDER BY run_id, epoch
"""

_VARS_LINE_RE = re.compile(r"^VARS\s+(\S+)\s+(.*)$")
_RUN_UUID_RE = re.compile(r'run_uuid="([0-9a-fA-F]+)"')
_PROTEIN_RE = re.compile(r'\bprotein="([^"]+)"')


def parse_protein_map(dag_path: Path) -> pd.DataFrame:
    """Parse a HTCondor DAG file's VARS lines into a run_id -> protein table.

    Each job's run_uuid and protein may appear on either of its two VARS
    lines, so this collects both keys per job name rather than assuming a
    fixed line order, then joins them by job name.

    Args:
        dag_path: Path to the .dag file.

    Returns:
        DataFrame with columns run_id, protein — one row per distinct run_uuid.

    Raises:
        ValueError: If a run_uuid is paired with more than one protein value.
    """
    job_run_uuid: dict[str, str] = {}
    job_protein: dict[str, str] = {}
    with open(dag_path) as f:
        for line in f:
            match = _VARS_LINE_RE.match(line)
            if not match:
                continue
            job_name, rest = match.groups()
            uuid_match = _RUN_UUID_RE.search(rest)
            if uuid_match:
                job_run_uuid[job_name] = uuid_match.group(1)
            protein_match = _PROTEIN_RE.search(rest)
            if protein_match:
                job_protein[job_name] = protein_match.group(1)

    pairs: dict[str, str] = {}
    conflicts: dict[str, set[str]] = {}
    for job_name, run_uuid in job_run_uuid.items():
        protein = job_protein.get(job_name)
        if protein is None:
            continue
        if run_uuid in pairs and pairs[run_uuid] != protein:
            conflicts.setdefault(run_uuid, {pairs[run_uuid]}).add(protein)
        pairs[run_uuid] = protein

    if conflicts:
        raise ValueError(f"run_uuid mapped to multiple proteins in {dag_path}: {conflicts}")

    return pd.DataFrame(sorted(pairs.items()), columns=["run_id", "protein"])


def load_metrics(db_path: Path) -> pd.DataFrame:
    """Read per-checkpoint metrics out of a (SQLite-backed) provenance db.

    Uses an explicit ATTACH ... (TYPE SQLITE, READ_ONLY) rather than a bare
    duckdb.connect(db_path) so the external-source boundary is visible in code
    and doesn't depend on version-specific file-format autodetection. The
    sqlite extension is installed/loaded explicitly so a missing/offline
    extension fails loudly here rather than mysteriously mid-query.
    """
    con = duckdb.connect(":memory:")
    con.install_extension("sqlite")
    con.load_extension("sqlite")
    con.execute(f"ATTACH '{db_path}' AS src (TYPE SQLITE, READ_ONLY)")
    con.execute("USE src")
    df = con.execute(METRICS_QUERY).df()
    con.close()
    return df


def load_run_protein_map(experiment: str) -> pd.DataFrame:
    """Read the pre-generated run_id -> protein lookup for one experiment.

    Regenerate with `python generate_protein_maps.py` if a source's DAG file
    changes — this reads the checked-in CSV, it does not parse the DAG file.
    """
    path = DATA_DIR / f"run_protein_map_{experiment}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python generate_protein_maps.py` first."
        )
    return pd.read_csv(path, dtype={"run_id": str, "protein": str})


def classify_training_strategy(metrics_df: pd.DataFrame) -> pd.Series:
    """Derive per-run training strategy from observed GPU model variance.

    A run is "mixed" if its checkpoints were produced on more than one GPU
    model, "device_constrained" if the GPU model never changed. This measures
    what actually happened rather than trusting a per-source label.
    """
    n_gpu_models = metrics_df.groupby("run_id")["gpu_model"].transform("nunique")
    return n_gpu_models.gt(1).map({True: "mixed", False: "device_constrained"})


@st.cache_data(ttl=300)
def _cached_load_metrics(db_path: str) -> pd.DataFrame:
    return load_metrics(Path(db_path))


@st.cache_data
def _cached_load_protein_map(experiment: str) -> pd.DataFrame:
    return load_run_protein_map(experiment)


@st.cache_data(ttl=300)
def get_dashboard_data() -> tuple[pd.DataFrame, int]:
    """Load and combine all configured SOURCES.

    Returns:
        Tuple of (combined dataframe, count of rows whose run_id had no entry
        in that source's protein map). Unmapped rows are kept with
        protein="unmapped" rather than silently dropped.
    """
    frames = []
    n_unmapped = 0
    for source in SOURCES:
        metrics = _cached_load_metrics(str(source["db_path"]))
        metrics["training_strategy"] = classify_training_strategy(metrics)
        metrics["experiment"] = source["experiment"]

        protein_map = _cached_load_protein_map(source["experiment"])
        merged = metrics.merge(protein_map, on="run_id", how="left")
        unmapped = merged["protein"].isna()
        n_unmapped += int(unmapped.sum())
        merged.loc[unmapped, "protein"] = "unmapped"

        frames.append(merged)

    combined = pd.concat(frames, ignore_index=True)
    return combined, n_unmapped
