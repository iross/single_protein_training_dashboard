set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list

# Install/update the uv-managed environment from pyproject.toml
sync:
    uv sync

# Launch the dashboard
run: sync
    uv run streamlit run app.py

# Regenerate run_protein_map_<experiment>.csv from each SOURCES entry's DAG file
protein-map: sync
    uv run python generate_protein_maps.py

# Headless smoke test: run app.py via Streamlit's AppTest API, no browser needed
check: sync
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
