FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

COPY pyproject.toml uv.lock app.py charts.py data.py build_dashboard_data.py ./
COPY .streamlit/ ./.streamlit/
COPY data/ ./data/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Bake the combined metrics table into the image so the running container
# never needs the raw databases or DuckDB at request time.
RUN uv run python build_dashboard_data.py

ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
