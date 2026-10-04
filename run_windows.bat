@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo [1/3] Installing Python libraries...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo pip install failed. Check that Python is installed.
  pause
  exit /b 1
)
echo [2/3] Checking Tesseract OCR...
where tesseract >nul 2>nul
if errorlevel 1 if not exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
  echo Installing Tesseract OCR with winget...
  winget install -e --id UB-Mannheim.TesseractOCR --accept-package-agreements --accept-source-agreements
)
echo [3/3] Starting the app...
python -m streamlit run app.py
pause
