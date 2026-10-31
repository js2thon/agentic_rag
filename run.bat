@echo off
setlocal
cd /d "%~dp0"

rem 端口: run.bat 8600 可覆盖默认值
set "PORT=%~1"
if "%PORT%"=="" set "PORT=8503"

rem 优先用项目内 venv: 全局 Python 可能装了同名无关包(如 agents 盖住 openai-agents)
if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

"%PY%" -X utf8 -m streamlit run app.py --server.port %PORT% --server.headless true
