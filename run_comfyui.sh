#!/usr/bin/env bash
#
# run_comfyui.sh
#
# Starts ComfyUI using the project virtual environment.
# Activates the venv and launches the server on port 8188.
#
# Usage:
#   ./run_comfyui.sh                         # start server (foreground)
#   ./run_comfyui.sh --listen 0.0.0.0        # bind all interfaces (LAN access)
#   ./run_comfyui.sh --port 8188 ...         # extra args passed to main.py
#
set -euo pipefail

cd "$(dirname "$0")"

# Local proxy for internet access (model downloads from HuggingFace, etc.).
# Adjust PROXY_HOST/PROXY_PORT if your proxy address changes.
PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
PROXY_PORT="${PROXY_PORT:-7890}"
export HTTP_PROXY="http://${PROXY_HOST}:${PROXY_PORT}"
export HTTPS_PROXY="http://${PROXY_HOST}:${PROXY_PORT}"
export ALL_PROXY="socks5://${PROXY_HOST}:${PROXY_PORT}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"

# Path to the virtual environment python.
PYTHON="${PYTHON:-./venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: venv python not found at $PYTHON"
  echo "Run: python3 -m venv venv && ./venv/bin/python -m pip install -r requirements.txt"
  exit 1
fi

echo "==> Starting ComfyUI with proxy ${HTTP_PROXY}"
exec "$PYTHON" main.py "$@"