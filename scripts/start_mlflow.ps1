# PowerShell launcher for MLflow Dashboard
$env:MLFLOW_ALLOW_FILE_STORE = "true"
$env:MLFLOW_DISABLE_AGENT_HINT = "1"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  STARTING MLFLOW DASHBOARD" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Opening MLflow UI at: http://localhost:5000" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the server.`n"

if (Test-Path ".\.venv\Scripts\mlflow.exe") {
    .\.venv\Scripts\mlflow.exe ui --port 5000 --backend-store-uri file:./mlruns
} elseif (Test-Path "..\.venv\Scripts\mlflow.exe") {
    ..\.venv\Scripts\mlflow.exe ui --port 5000 --backend-store-uri file:../mlruns
} else {
    mlflow ui --port 5000 --backend-store-uri file:./mlruns
}

