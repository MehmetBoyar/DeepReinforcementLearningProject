@echo off
TITLE Traffic RL Control Dashboard
CLS

:: Ensure we are running in the script's directory
CD /D "%~dp0"

ECHO ========================================================
ECHO      TRAFFIC RL RESEARCH DASHBOARD
ECHO ========================================================
ECHO.

:: Check if Python is available
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO [ERROR] Python is not installed or not added to PATH.
    ECHO Please install Python 3.8+ and try again.
    PAUSE
    EXIT /B
)

::  Check for Virtual Environment
IF NOT EXIST "venv" (
    ECHO [SETUP] Virtual environment 'venv' not found.
    ECHO [SETUP] Creating environment...
    python -m venv venv
    
    ECHO [SETUP] Activating environment...
    CALL venv\Scripts\activate.bat
    
    ECHO [SETUP] Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    
    ECHO [SETUP] Installation complete!
    ECHO.
) ELSE (
    ECHO [INFO] Virtual environment found. Activating...
    CALL venv\Scripts\activate.bat
)

:: Launch the Application
ECHO.
ECHO [START] Launching Streamlit...
ECHO [INFO]  Press Ctrl+C in this window to stop the server.
ECHO.

streamlit run Traffic_RL.py

:: Pause if it crashes so you can read the error
IF %ERRORLEVEL% NEQ 0 (
    ECHO.
    ECHO ========================================================
    ECHO [ERROR] The application closed unexpectedly.
    ECHO ========================================================
    PAUSE
)