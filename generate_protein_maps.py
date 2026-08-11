"""One-off script: parse each configured DAG file into a run_id -> protein CSV.

Run this whenever a source's DAG file is added or changes:

    uv run python generate_protein_maps.py

The dashboard reads the generated CSVs at runtime; it never parses DAG files
directly (see data.load_run_protein_map).
"""

from data import SOURCES, parse_protein_map


def main() -> None:
    for source in SOURCES:
        df = parse_protein_map(source["dag_path"])
        out_path = source["dag_path"].parent / f"run_protein_map_{source['experiment']}.csv"
        df.to_csv(out_path, index=False)
        print(f"{source['experiment']}: wrote {len(df)} run_id -> protein rows to {out_path}")


if __name__ == "__main__":
    main()
