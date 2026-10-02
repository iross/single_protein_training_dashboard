# Single-protein training dashboard

Streamlit dashboard for per-protein test_loss / pearson_total_score training metrics.

## Development

```
just run    # launch the dashboard locally
just check  # headless smoke test
```

## Data layout

All source data lives under `data/`:

- `provenance.db`, `mixed.db`, `dgxspark.db`, `metl_updates.db` — per-checkpoint metrics (read
  at build time by `build_dashboard_data.py`, which queries them via DuckDB's SQLite
  extension).
- `*.dag` — HTCondor DAG files, parsed only by `generate_protein_maps.py` to regenerate the
  run_id -> protein CSVs below. Not read at build or app runtime. A source can have several
  DAGs (e.g. a `_run2.dag` rerun); `just update-data` pulls all of them from ap2002.
- `protein_map_*.csv` — hand-written run_id -> protein lookups for runs whose DAG file no
  longer exists (listed under a source's `extra_protein_maps`), merged in by
  `generate_protein_maps.py`.
- `run_protein_map_*.csv` — pre-generated run_id -> protein lookups, read by
  `build_dashboard_data.py`. Regenerate with `just protein-map` after changing a source's
  DAG files.
- `dashboard_data.csv` — the combined, precomputed table the app actually reads
  (`data.py`). Regenerate with `just build-data` after changing a source's database.

The app itself (`app.py`, `charts.py`, `data.py`) only ever reads `dashboard_data.csv` — it
has no DuckDB/SQLite dependency at request time. `Dockerfile` runs
`build_dashboard_data.py` at image build time, so the deployed container never needs the
raw databases either.

## Deployment

The app runs as a normal containerized Streamlit server, deployed to a Kubernetes cluster
via GitOps — no third-party hosting account or OAuth app involved.

- `Dockerfile` builds an image with `dashboard_data.csv` baked in (see above). Test it
  locally with `docker build -t single-protein-training-dashboard . && docker run -p
  8501:8501 single-protein-training-dashboard`.
- `.github/workflows/build-image.yml` builds and pushes the image to
  `hub.osg-htc.org/xdd/single_protein_training_dashboard` on every push to `main`, tagged
  `latest` and with the git short SHA. Push credentials come from the `HARBOR_USERNAME` /
  `HARBOR_PASSWORD` repo secrets (a Harbor robot account).
- `k8s/deployment.yaml` is a reference Deployment/Service/Ingress — copy it into the
  cluster's GitOps repo (adjust namespace, ingress host/class, resource limits) and let
  Argo CD/Flux reconcile it. This repo doesn't push to the GitOps repo itself.
- `.github/workflows/ci.yml` runs the headless smoke test (`just check`) on every push and
  PR to `main`.

Getting a new image rolled out is two steps: this repo's workflow publishes the image, then
the GitOps controller needs to pick up the new tag — either via image automation (Argo CD
Image Updater / Flux image-reflector) watching the registry, or by bumping the tag in the
manifests repo by hand.

## Updating data

**Pulling fresh databases from ap2002**: the `mldag-query db build` commands in
`data_compilation.md` regenerate each source's `.db` from its checkpoint directory on
ap2002. `just update-data` runs them over SSH and scps the results into `data/`:

```
just update-data  # ssh to ap2002, rebuild every source .db, scp them and the DAGs down
just build-data   # regenerates data/dashboard_data.csv from the new .db files
just check        # confirms the app still renders with the new data
```

**Refreshing existing sources by hand** (new checkpoints appended to the same databases):
replace `data/provenance.db`, `data/mixed.db`, `data/dgxspark.db`, and/or
`data/metl_updates.db` with updated copies, then:

```
just build-data   # regenerates data/dashboard_data.csv from the new .db files
just check        # confirms the app still renders with the new data
```

**Adding a new experiment/source**:

1. Drop the new `.db` and `.dag` files into `data/`.
2. Add an entry to `SOURCES` in `build_dashboard_data.py` (db_path, dag_paths, experiment name).
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

Merging to `main` rebuilds and pushes a new image with the updated data (see Deployment) —
but the running deployment won't update until the GitOps controller rolls out the new tag.
