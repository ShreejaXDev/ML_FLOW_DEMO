@echo off
echo ======================================================================
echo  MLFLOW EXPERIMENT TRACKING & MODEL COMPARISON: FRAUD DETECTION
echo ======================================================================
echo.

cd /d "%~dp0\.."

REM Activate virtual environment if present
if exist ".venv\Scripts\activate.bat" (
    echo Activating Python virtual environment (.venv)...
    call .venv\Scripts\activate.bat
) else (
    echo Virtual environment .venv not found. Using system python...
)

echo.
echo Running full ML training and evaluation pipeline...
echo ======================================================================
set MLFLOW_ALLOW_FILE_STORE=true
set MLFLOW_DISABLE_AGENT_HINT=1

if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe main.py
) else (
    python main.py
)
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Pipeline execution failed with exit code %ERRORLEVEL%.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ======================================================================
echo Pipeline completed successfully!
echo To view results in the MLflow Dashboard, run:
echo    scripts\start_mlflow.bat
echo or:
echo    mlflow ui --port 5000
echo ======================================================================
pause

