@echo off
echo.
echo ========================================
echo   Empathetic Friend Chat Server
echo ========================================
echo.

REM Set model path (change this if your model is elsewhere)
set MODEL_PATH=..\trained_model\lora_adapter

echo Model Path: %MODEL_PATH%
echo.

REM Check if model exists
if not exist "%MODEL_PATH%" (
    echo ERROR: Model not found at %MODEL_PATH%
    echo.
    echo Please extract your model first:
    echo   cd ..\trained_model
    echo   tar -xf empathetic_model.zip
    echo.
    pause
    exit /b 1
)

echo Starting server on http://localhost:8000
echo Press Ctrl+C to stop
echo.

python -m uvicorn app:app --host 0.0.0.0 --port 8000

pause

