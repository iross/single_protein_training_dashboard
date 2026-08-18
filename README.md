# Single-protein training dashboard

Streamlit dashboard for per-protein test_loss / pearson_total_score training metrics.

## Development

```
just run    # launch the dashboard locally
just check  # headless smoke test
```

## Data layout

All source data lives under `data/`:

- `provenance.db`, `mixed.db` — per-checkpoint metrics (DuckDB reads these SQLite files at
  query time via `data.py`).
- `*.dag` — HTCondor DAG files, parsed only by `generate_protein_maps.py` to regenerate the
  run_id -> protein CSVs below. Not read at app runtime.
- `run_protein_map_*.csv` — pre-generated run_id -> protein lookups, read by the app at
  runtime. Regenerate with `just protein-map` after changing a source's DAG file.

## Deployment

The app is hosted on [Streamlit Community Cloud](https://share.streamlit.io), connected to
this repo's `main` branch. Community Cloud watches the branch directly and redeploys
automatically on every push — no GitHub Action is needed for the deploy step itself.

`.github/workflows/ci.yml` runs the headless smoke test (`just check`) on every push and PR
to `main`, so breakage is caught before it reaches the live app.

One-time setup for a new deployment target: go to share.streamlit.io, connect this repo, and
point it at `app.py` on `main`.
