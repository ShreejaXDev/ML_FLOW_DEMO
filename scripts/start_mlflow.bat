@echo off
echo ======================================================================
echo  STARTING MLFLOW DASHBOARD
echo ======================================================================
echo.

cd /d "%~dp0\.."

REM Activate virtual environment if present
if exist ".venv\Scripts\activate.bat" (
    echo Activating Python virtual environment (.venv)...
    call .venv\Scripts\activate.bat
)

echo.
echo Opening MLflow UI on http://localhost:5000...
echo Tracking Directory: %CD%\mlruns
echo.
echo Press Ctrl+C in this terminal to stop the MLflow server.
echo ======================================================================

set MLFLOW_ALLOW_FILE_STORE=true
set MLFLOW_DISABLE_AGENT_HINT=1

if exist ".venv\Scripts\mlflow.exe" (
    .venv\Scripts\mlflow.exe ui --port 5000 --backend-store-uri file:./mlruns
) else (
    mlflow ui --port 5000 --backend-store-uri file:./mlruns
)
pause

