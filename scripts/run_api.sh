#!/bin/bash
cd "$(dirname "$0")/.."
pip install fastapi uvicorn --break-system-packages 2>/dev/null
python -m uvicorn virtual_cell.api:app --host 0.0.0.0 --port 8099 --reload
