#!/bin/bash
# 启动 TypoAgent Web Tool
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PYTHONPATH"
pip install -q flask flask-cors 2>/dev/null || true
python -m tool.app
