#!/bin/sh
# -----------------------------------------------------------------------------
# Container entrypoint.
# Logs the resolved ONNX Runtime providers before starting uvicorn so it's
# obvious from `docker logs` whether GPU is active.
# -----------------------------------------------------------------------------
set -eu

python - <<'PY'
import onnxruntime as ort
print(f"[entrypoint] onnxruntime version: {ort.__version__}")
print(f"[entrypoint] available providers: {ort.get_available_providers()}")
PY

exec "$@"
