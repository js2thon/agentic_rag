#!/usr/bin/env bash
# 启动 Streamlit 应用: ./run.sh [端口]
set -euo pipefail
cd "$(dirname "$0")"

PORT="${1:-8503}"

# 优先用项目内 venv
if [ -x ".venv/Scripts/python.exe" ]; then
  PY=".venv/Scripts/python.exe"   # Windows(Git Bash)
elif [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

exec "$PY" -X utf8 -m streamlit run app.py --server.port "$PORT" --server.headless true
