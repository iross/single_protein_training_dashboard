"""One-off script: combine each source's DAG files (and any extra maps) into a protein CSV.

Run this whenever a source's DAG files are added or change:

    uv run python generate_protein_maps.py

build_dashboard_data.py reads the generated CSVs; it never parses DAG files
directly (see build_dashboard_data.load_run_protein_map).
"""

import pandas as pd

from build_dashboard_data import DATA_DIR, SOURCES, parse_protein_map


def combine_protein_maps(maps: list[pd.DataFrame], experiment: str) -> pd.DataFrame:
    """Concatenate per-DAG maps, failing if a run_id maps to different proteins."""
    combined = pd.concat(maps, ignore_index=True).drop_duplicates()
    conflicts = combined[combined["run_id"].duplicated(keep=False)]
    if not conflicts.empty:
        raise ValueError(f"{experiment}: run_id mapped to multiple proteins:\n{conflicts}")
    return combined.sort_values("run_id", ignore_index=True)


def main() -> None:
    for source in SOURCES:
        maps = [parse_protein_map(dag_path) for dag_path in source["dag_paths"]]
        maps += [
            pd.read_csv(path, dtype=str, usecols=["run_id", "protein"])
            for path in source.get("extra_protein_maps", [])
        ]
        df = combine_protein_maps(maps, source["experiment"])
        out_path = DATA_DIR / f"run_protein_map_{source['experiment']}.csv"
        df.to_csv(out_path, index=False)
        print(f"{source['experiment']}: wrote {len(df)} run_id -> protein rows to {out_path}")


if __name__ == "__main__":
    main()
