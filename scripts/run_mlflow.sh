#!/usr/bin/env bash
# Launch the MLflow UI over our training runs.
#
# MLflow runs in its own isolated venv (.venv-mlflow) so its dependencies never
# touch the main project (torch/streamlit). The first run creates that venv and
# ingests the runs from logs/ into a local sqlite store; afterwards it just
# re-ingests (cheap) and serves the UI.
#
#   bash scripts/run_mlflow.sh          # → http://127.0.0.1:5000
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x .venv-mlflow/bin/mlflow ]; then
  echo "· Creating isolated MLflow environment (.venv-mlflow)…"
  uv venv .venv-mlflow --python 3.12
  uv pip install --python .venv-mlflow "mlflow>=2.16" pandas
fi

echo "· Ingesting runs from logs/ into ./mlflow.db…"
.venv-mlflow/bin/python scripts/ingest_mlflow.py

echo "· Starting MLflow UI at http://127.0.0.1:5000  (Ctrl+C to stop)"
echo "  Tip: switch the top-left toggle to 'Model training', then open the experiment."
exec .venv-mlflow/bin/mlflow ui --backend-store-uri "sqlite:///$(pwd)/mlflow.db" \
  --host 127.0.0.1 --port 5000
