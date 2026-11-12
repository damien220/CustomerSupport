# Multi-stage build. Lite image by default; INSTALL_RAG=1 adds the torch stack. No internal wheels are needed (api_client_base was
# replaced by the LLM gateway in Phase 3).

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# torch (pulled in transitively by sentence-transformers in requirements-rag.txt)
# has no plain PyPI wheel for every platform/Python combo and defaults to a huge
# CUDA-bundled build on linux/amd64 even when no GPU is present. Install it
# explicitly first from PyTorch's own index so the build is reproducible and as
# small as possible; override TORCH_INDEX_URL at build time for a CUDA image,
# e.g. --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
# (see https://pytorch.org/get-started/locally/ for the full index list).
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# The lite core (default) needs no torch. Build with --build-arg INSTALL_RAG=1 to add the
# heavy extras (sentence-transformers / BGE reranker / Gradio / RAGAS) — required for
# EMBED_PROVIDER=local, RERANKER=bge and --mode ui.
ARG INSTALL_RAG=0

COPY requirements.txt requirements-rag.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "${INSTALL_RAG}" = "1" ]; then \
         pip install --no-cache-dir torch --index-url "${TORCH_INDEX_URL}" \
         && pip install --no-cache-dir -r requirements-rag.txt; \
       fi

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime
WORKDIR /app

RUN groupadd --system appgroup && useradd --system --gid appgroup appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY app.py bootstrap.py server.py remote_mcp.py ./
COPY faq_core/ ./faq_core/
COPY api/ ./api/
COPY background/ ./background/
COPY formatters/ ./formatters/
COPY observability/ ./observability/
COPY ui/ ./ui/
# Seed FAQ + retrieval benchmark (databases are excluded by .dockerignore).
COPY data/sample_faq.json data/retrieval_benchmark.json ./data/

RUN mkdir -p /app/data && chown -R appuser:appgroup /app
USER appuser

EXPOSE 8000 8001 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["python", "app.py"]
CMD ["--mode", "api"]
