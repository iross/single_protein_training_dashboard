"""One-off script: parse each configured DAG file into a run_id -> protein CSV.

Run this whenever a source's DAG file is added or changes:

    uv run python generate_protein_maps.py

build_dashboard_data.py reads the generated CSVs; it never parses DAG files
directly (see build_dashboard_data.load_run_protein_map).
"""

from build_dashboard_data import SOURCES, parse_protein_map


def main() -> None:
    for source in SOURCES:
        if source["dag_path"] is None:
            continue
        df = parse_protein_map(source["dag_path"])
        out_path = source["dag_path"].parent / f"run_protein_map_{source['experiment']}.csv"
        df.to_csv(out_path, index=False)
        print(f"{source['experiment']}: wrote {len(df)} run_id -> protein rows to {out_path}")


if __name__ == "__main__":
    main()
