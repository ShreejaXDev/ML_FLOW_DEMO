"""
Launcher for MLflow Web Dashboard.

Automatically configures the local file-store environment variables
and launches the interactive dashboard on port 5000.
"""

import os
import sys

# Opt into local file-based tracking store (mlruns/)
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

if __name__ == "__main__":
    print("=" * 70)
    print("  STARTING MLFLOW DASHBOARD")
    print("=" * 70)
    print("Tracking Store: ./mlruns")
    print("URL           : http://localhost:5000")
    print("Press Ctrl+C to stop the server.")
    print("=" * 70 + "\n")

    from mlflow.cli import cli
    sys.argv = ["mlflow", "ui", "--port", "5000", "--backend-store-uri", "file:./mlruns"]
    cli()

