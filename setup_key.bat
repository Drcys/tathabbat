@echo off
cd /d "%~dp0"
echo.
echo  Enable reading of decorative Arabic scripts (Thuluth, Diwani, Kufi, handwriting)
echo  ---------------------------------------------------------------------------
echo  1) A Google AI Studio page will open. Sign in, click "Create API key", copy it.
echo  2) Come back here, paste the key (right-click or Ctrl+V), then press Enter.
echo.
start "" "https://aistudio.google.com/app/apikey"
set "KEY="
set /p KEY=Gemini API key: 
if not defined KEY (
  echo No key entered.
  pause
  exit /b 1
)
if not exist ".streamlit" mkdir ".streamlit"
> ".streamlit\secrets.toml" echo GEMINI_API_KEY = "%KEY%"
echo.
echo  Saved to .streamlit\secrets.toml  (this file is never uploaded to GitHub).
echo  Now start the site with run_windows.bat
pause
