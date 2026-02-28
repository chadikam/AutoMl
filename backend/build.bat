@echo off
REM ─────────────────────────────────────────────────────────────────
REM  AutoML Framework — Build Standalone Windows Executable
REM ─────────────────────────────────────────────────────────────────
setlocal

echo ============================================
echo   AutoML Framework — Build Script
echo ============================================
echo.

REM ── 1. Build frontend ──────────────────────────────────────────
echo [1/4] Building frontend...
pushd ..\frontend
call npm install
if %ERRORLEVEL% neq 0 (
    echo ERROR: npm install failed
    popd
    exit /b 1
)
call npm run build
if %ERRORLEVEL% neq 0 (
    echo ERROR: frontend build failed
    popd
    exit /b 1
)
popd
echo       Frontend built successfully.
echo.

REM ── 2. Install PyInstaller if needed ───────────────────────────
echo [2/4] Ensuring PyInstaller is installed...
pip show pyinstaller >nul 2>&1
if %ERRORLEVEL% neq 0 (
    pip install pyinstaller
)
echo       PyInstaller ready.
echo.

REM ── 3. Run PyInstaller ─────────────────────────────────────────
echo [3/4] Building executable with PyInstaller...
python -m PyInstaller automl.spec --clean --noconfirm
if %ERRORLEVEL% neq 0 (
    echo ERROR: PyInstaller build failed
    exit /b 1
)
echo       Executable built successfully.
echo.

REM ── 4. Summary ─────────────────────────────────────────────────
echo [4/4] Build complete!
echo.
echo   Output folder:  dist\AutoML\
echo   Executable:     dist\AutoML\AutoML.exe
echo.
echo   To run:
echo     cd dist\AutoML
echo     AutoML.exe
echo.
echo   The app will start on http://127.0.0.1:8000
echo   and auto-open your browser.
echo ============================================

endlocal
