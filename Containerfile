# One image for api, worker, ingest and the replay CLI.
FROM docker.io/library/node:22-slim AS web
WORKDIR /web
# analytics packages pulled in by @patternfly/chatbot are overridden with an empty local stub (no telemetry code shipped)
COPY frontend/package.json frontend/package-lock.json frontend/.npmrc ./
COPY frontend/stubs ./stubs
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM docker.io/library/python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONPATH=/app/backend MPLCONFIGDIR=/tmp HOME=/tmp CUPY_CACHE_DIR=/tmp/cupy \
    JFE_PACK=/app/data/processed JFE_MANIFEST=/app/data/manifests/sources.json JFE_STATIC=/app/frontend/dist JFE_ARTIFACTS=/data/artifacts
WORKDIR /app
COPY backend/pyproject.toml backend/uv.lock backend/
RUN cd backend && uv sync --frozen --no-install-project --extra gpu
# Data pack: downloaded from opendata.gov.je and verified against the pinned SHA-256 manifest.
COPY data/manifests data/manifests
COPY backend/jfe/__init__.py backend/jfe/ingest.py backend/jfe/
RUN python -m jfe.ingest --raw /app/data/raw --out /app/data/processed
COPY backend/ backend/
COPY --from=web /web/dist frontend/dist
ARG GIT_SHA=unknown
ENV JFE_GIT_SHA=$GIT_SHA JFE_IMAGE=localhost/jfe-app:$GIT_SHA
RUN useradd -r -u 10001 jfe && mkdir -p /data/artifacts && chown jfe /data/artifacts && chmod -R a+rX /app
USER jfe
EXPOSE 8000
CMD ["uvicorn", "jfe.api:app", "--host", "0.0.0.0", "--port", "8000"]
