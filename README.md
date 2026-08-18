# Single-protein training dashboard

Streamlit dashboard for per-protein test_loss / pearson_total_score training metrics.

## Development

```
just run    # launch the dashboard locally
just check  # headless smoke test
```

## Data layout

All source data lives under `data/`:

- `provenance.db`, `mixed.db` — per-checkpoint metrics (read at build time by
  `build_dashboard_data.py`, which queries them via DuckDB's SQLite extension).
- `*.dag` — HTCondor DAG files, parsed only by `generate_protein_maps.py` to regenerate the
  run_id -> protein CSVs below. Not read at build or app runtime.
- `run_protein_map_*.csv` — pre-generated run_id -> protein lookups, read by
  `build_dashboard_data.py`. Regenerate with `just protein-map` after changing a source's
  DAG file.
- `dashboard_data.csv` — the combined, precomputed table the app actually reads
  (`data.py`). Regenerate with `just build-data` after changing a source's database.

The app itself (`app.py`, `charts.py`, `data.py`) only ever reads `dashboard_data.csv` — it
has no DuckDB/SQLite dependency. That's what lets it run unmodified in the browser via
stlite (see Deployment), which can't load DuckDB's SQLite extension.

## Deployment

The app is published as a static site on GitHub Pages using
[stlite](https://github.com/whitphx/stlite), which runs the real Streamlit app client-side
in the browser via Pyodide (WASM) — no server, no third-party hosting account, no OAuth app
installed on this GitHub account.

`.github/workflows/deploy-pages.yml` runs on every push to `main`: it rebuilds
`dashboard_data.csv` from the databases, assembles `index.html` + `app.py` + `charts.py` +
`data.py` + `data/dashboard_data.csv` into a `dist/` directory, and deploys it via GitHub's
own `actions/deploy-pages` (authenticated with the workflow's built-in token, not an external
app).

`.github/workflows/ci.yml` runs the headless smoke test (`just check`) on every push and PR
to `main`.

One-time setup for a new repo: Settings → Pages → Source → "GitHub Actions".

## Updating data

**Refreshing existing sources** (new checkpoints appended to the same databases): replace
`data/provenance.db` and/or `data/mixed.db` with updated copies, then:

```
just build-data   # regenerates data/dashboard_data.csv from the new .db files
just check        # confirms the app still renders with the new data
```

**Adding a new experiment/source**:

1. Drop the new `.db` and `.dag` files into `data/`.
2. Add an entry to `SOURCES` in `build_dashboard_data.py` (db_path, dag_path, experiment name).
3. `just protein-map` — parses the new DAG into `data/run_protein_map_<experiment>.csv`.
4. `just build-data && just check` to verify.

**Publishing**: direct pushes to `main` are blocked, so go through a branch + PR:

```
git checkout -b update-data
git add data/ build_dashboard_data.py   # whatever changed
git commit -m "..."
git push -u origin update-data
gh pr create --fill
gh pr merge --merge --delete-branch     # after CI is green
```

Merging to `main` is the trigger — `deploy-pages.yml` fires automatically, rebuilds
`dashboard_data.csv` from whatever `.db` files are on `main`, and republishes to the same
URL. No manual deploy step.
