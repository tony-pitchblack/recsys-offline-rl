#!/usr/bin/env bash
set -a; source .env; set +a; \
PORT_CLEAN="$(printf '%s' "$MLFLOW_PORT" | tr -d "\"'")"; \
HOST_CLEAN="$(printf '%s' "$MLFLOW_HOST" | tr -d "\"'")"; \
DISPLAY_HOST="$HOST_CLEAN"; \
if [ -z "$DISPLAY_HOST" ] || [ "$DISPLAY_HOST" = "0.0.0.0" ] || [ "$DISPLAY_HOST" = "::" ]; then DISPLAY_HOST="localhost"; fi; \
SESSION="mlflow-${PORT_CLEAN}"; \
ROOT="$PWD/data/mlflow"; \
DB_PATH="$PWD/data/mlflow.db"; \
MLFLOW_BIN="$(command -v mlflow || true)"; \
if [ -z "$MLFLOW_BIN" ]; then \
  echo "ERROR: mlflow is not available in current shell."; \
  echo "Activate your venv first:"; \
  echo "source .venv/bin/activate"; \
  exit 1; \
fi; \
mkdir -p "$ROOT"/{artifacts,}; \
tmux has-session -t "$SESSION" 2>/dev/null && { echo "http://${DISPLAY_HOST}:${PORT_CLEAN}"; echo "tmux attach -t ${SESSION}"; exit 0; }; \
tmux new-session -d -s "$SESSION" "bash -ic 'cd \"$PWD\"; source .env; exec \"$MLFLOW_BIN\" server \
  --host \"$HOST_CLEAN\" \
  --port \"$PORT_CLEAN\" \
  --backend-store-uri \"sqlite:///$DB_PATH\" \
  --default-artifact-root \"file://$ROOT/artifacts\" \
  --serve-artifacts \
  --workers 1'"
echo "http://${DISPLAY_HOST}:${PORT_CLEAN}"
echo "tmux attach -t ${SESSION}"
