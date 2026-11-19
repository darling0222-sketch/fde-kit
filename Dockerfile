FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FDE_HOME=/app/data \
    FDE_CORPUS=/app/examples/corpus \
    FDE_ONTOLOGY=/app/ontology/after_sales.yaml

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[docs]"

COPY ontology ./ontology
COPY evals ./evals
COPY examples ./examples

RUN mkdir -p /app/data

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

CMD ["fde", "serve", "--host", "0.0.0.0", "--port", "8000"]
