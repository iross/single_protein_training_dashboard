set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list

# Install/update the uv-managed environment from pyproject.toml
sync:
    uv sync

# Launch the dashboard
run: sync
    uv run streamlit run app.py

# Regenerate run_protein_map_<experiment>.csv from each SOURCES entry's DAG files
protein-map: sync
    uv run python generate_protein_maps.py

# Regenerate protein maps, then rebuild data/dashboard_data.csv from the raw databases
build-data: protein-map
    uv run python build_dashboard_data.py

# Render device_constrained vs mixed report figures and summary tables into figures/report/
report-figures: sync
    uv run python make_report_figures.py

# Rebuild the raw provenance databases on ap2002 (see data_compilation.md), scp them and the DAGs down
update-data:
    ssh ap2002.chtc.wisc.edu ' \
        set -euo pipefail && \
        export PATH="$HOME/.local/bin:$PATH" && \
        cd ~/single_protein_models_gpu_device_constrained && \
        uv run mldag-query db build --checkpoint-dir /staging/i/iaross/single_protein_models_gpu_device_constrained --db provenance.db && \
        cd ~/single_protein_models_with_ospool && \
        uv run mldag-query db build --checkpoint-dir /staging/i/iaross/single_protein_checkpoints_with_ospool --db mixed.db && \
        cd ~/single_protein_models_dgxspark && \
        uv run mldag-query db build --checkpoint-dir /staging/i/iaross/single_protein_models_dgxspark --db dgxspark.db && \
        cd ~/single_protein_models_metl_updates && \
        uv run mldag-query db build --checkpoint-dir /staging/i/iaross/single_protein_models_metl_updates --db metl_updates.db \
    '
    scp ap2002.chtc.wisc.edu:~/single_protein_models_gpu_device_constrained/provenance.db data/provenance.db
    scp ap2002.chtc.wisc.edu:~/single_protein_models_with_ospool/mixed.db data/mixed.db
    scp ap2002.chtc.wisc.edu:~/single_protein_models_dgxspark/dgxspark.db data/dgxspark.db
    scp ap2002.chtc.wisc.edu:~/single_protein_models_metl_updates/metl_updates.db data/metl_updates.db
    scp 'ap2002.chtc.wisc.edu:~/single_protein_models_gpu_device_constrained/*.dag' data/
    scp 'ap2002.chtc.wisc.edu:~/single_protein_models_with_ospool/*.dag' data/
    scp 'ap2002.chtc.wisc.edu:~/single_protein_models_dgxspark/*.dag' data/
    scp 'ap2002.chtc.wisc.edu:~/single_protein_models_metl_updates/*.dag' data/

# Headless smoke test: rebuild data, then run app.py via Streamlit's AppTest API
check: build-data
    uv run python -c "\
    from streamlit.testing.v1 import AppTest; \
    at = AppTest.from_file('app.py', default_timeout=60); \
    at.run(); \
    assert not at.exception, at.exception; \
    print(f'OK: {len(at.get(\"plotly_chart\"))} charts rendered, no exceptions')"

# Remove the virtualenv and cached bytecode
clean:
    rm -rf .venv
    find . -name __pycache__ -type d -exec rm -rf {} +
