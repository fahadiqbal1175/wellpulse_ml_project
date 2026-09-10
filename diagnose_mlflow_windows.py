"""Read-only diagnostic — prints what's stored in mlflow.db for the
Production model version versus what actually exists on disk, so we
can pinpoint the exact mismatch instead of guessing."""
import os
from urllib.parse import urlparse, unquote

import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

REGISTERED_MODEL_NAME = "wellpulse_final_model"

mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "Production")
print("Model version:", mv.version)
print("Model version source (raw):", repr(mv.source))
print("Model version run_id:", mv.run_id)

run = client.get_run(mv.run_id)
print("\nRun artifact_uri (raw):", repr(run.info.artifact_uri))

exp = client.get_experiment(run.info.experiment_id)
print("Experiment artifact_location (raw):", repr(exp.artifact_location))

for label, uri in [("model version source", mv.source), ("run artifact_uri", run.info.artifact_uri)]:
    parsed = urlparse(uri)
    print(f"\n--- {label} ---")
    print("  scheme:", parsed.scheme)
    print("  parsed.path (raw):", repr(parsed.path))
    decoded = unquote(parsed.path)
    print("  decoded path:", repr(decoded))
    # Windows file:// URIs often parse with a leading slash before the
    # drive letter (e.g. "/D:/foo") which is NOT a valid Windows path.
    candidates = [decoded, decoded.lstrip("/"), decoded.replace("/", "\\").lstrip("\\")]
    for c in candidates:
        print(f"  exists? {os.path.exists(c)!s:5}  ->  {c}")

print("\n--- what's actually under mlruns/ (first 3 levels) ---")
for root, dirs, files in os.walk("mlruns"):
    depth = root.replace("mlruns", "").count(os.sep)
    if depth > 2:
        dirs[:] = []
        continue
    print(root, "-> files:", files[:5])
