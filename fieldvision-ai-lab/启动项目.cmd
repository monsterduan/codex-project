@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please follow README.md to create .venv and install the project.
  pause
  exit /b 1
)
echo Open http://127.0.0.1:8765 in your browser.
".venv\Scripts\python.exe" -m fieldvision.cli serve
pause
