"""Build dashboard_data.csv from the raw provenance databases and DAG files.

This is the hotswap boundary: adding an experiment or pointing at a different
database means editing SOURCES here, not touching data.py, charts.py, or
app.py. The app itself never reads these raw sources directly — it only
reads the CSV this script produces, so it runs the same way locally and in
the browser (stlite/Pyodide) without needing DuckDB or SQLite there.

Run this whenever a source's database or DAG file changes:

    uv run python build_dashboard_data.py
"""

import re
from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).parent
DATA_DIR = REPO_ROOT / "data"

# dag_paths lists every DAG submitted for a source (e.g. a _run2.dag rerun);
# their run_id -> protein maps are combined into one CSV per experiment.
# extra_protein_maps lists hand-written run_id,protein CSVs for runs whose DAG
# file no longer exists. A fixed training_strategy labels every run in the
# source instead of classifying runs by GPU model variance.
SOURCES = [
    {
        "db_path": DATA_DIR / "provenance.db",
        "dag_paths": [
            DATA_DIR
            / "many_protein_pretraining_with_ospool_device_constrained_runs.dag",
            DATA_DIR
            / "many_protein_pretraining_with_ospool_device_constrained_runs_run2.dag",
        ],
        # June 2026 avgfp runs (DAGMan job 5424141) whose DAG was overwritten;
        # recovered from the jobs' HTCondor Args.
        "extra_protein_maps": [DATA_DIR / "protein_map_device_constrained_june.csv"],
        "experiment": "device_constrained",
    },
    {
        "db_path": DATA_DIR / "mixed.db",
        "dag_paths": [
            DATA_DIR / "many_protein_pretraining_with_ospool.dag",
            DATA_DIR / "many_protein_pretraining_with_ospool_run2.dag",
        ],
        "experiment": "heterogeneous",
    },
    {
        "db_path": DATA_DIR / "dgxspark.db",
        "dag_paths": [DATA_DIR / "many_protein_pretraining_dgx_spark.dag"],
        "experiment": "dgxspark",
        "training_strategy": "dgx_spark",
    },
    {
        "db_path": DATA_DIR / "metl_updates.db",
        # Same training setup rerun with updated software libraries.
        "dag_paths": [DATA_DIR / "many_protein_pretraining_updated_metl.dag"],
        "experiment": "metl_updates",
        "training_strategy": "updated_metl",
    },
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
        checkpoint_path,
        produced_at_ts
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
        raise ValueError(
            f"run_uuid mapped to multiple proteins in {dag_path}: {conflicts}"
        )

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


TEST_METRICS = ["test_loss", "pearson_total_score"]


def carry_forward_test_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Fill each run's test metrics forward from its last logged value.

    Test metrics are only logged when val_loss reaches a new best, so most
    late-epoch checkpoints have none. After filling, a checkpoint's test metrics
    are those of the run's best-validation checkpoint so far -- the model
    training would select -- and every run contributes at every epoch.
    test_metrics_logged marks the checkpoints whose values were logged directly.
    """
    ordered = metrics_df.sort_values(["run_id", "epoch"])
    logged = ordered["test_loss"].notna()
    filled = ordered.groupby("run_id")[TEST_METRICS].ffill()
    return ordered.assign(
        **{m: filled[m] for m in TEST_METRICS}, test_metrics_logged=logged
    )


def build_dashboard_data() -> tuple[pd.DataFrame, int]:
    """Load and combine all configured SOURCES.

    Returns:
        Tuple of (combined dataframe, count of rows whose run_id had no entry
        in that source's protein map). Unmapped rows are kept with
        protein="unmapped" rather than silently dropped.
    """
    frames = []
    n_unmapped = 0
    for source in SOURCES:
        metrics = carry_forward_test_metrics(load_metrics(source["db_path"]))
        metrics["training_strategy"] = source.get(
            "training_strategy", classify_training_strategy(metrics)
        )
        metrics["experiment"] = source["experiment"]

        protein_map = load_run_protein_map(source["experiment"])
        merged = metrics.merge(protein_map, on="run_id", how="left")
        unmapped = merged["protein"].isna()
        n_unmapped += int(unmapped.sum())
        merged.loc[unmapped, "protein"] = "unmapped"

        frames.append(merged)

    combined = pd.concat(frames, ignore_index=True)
    return combined, n_unmapped


def main() -> None:
    combined, n_unmapped = build_dashboard_data()
    out_path = DATA_DIR / "dashboard_data.csv"
    combined.to_csv(out_path, index=False)
    print(f"Wrote {len(combined)} rows ({n_unmapped} unmapped) to {out_path}")


if __name__ == "__main__":
    main()
