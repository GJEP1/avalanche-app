#!/bin/bash
# Launch Rock Avalanche Runout Explorer
# Usage: ./launch_explorer.sh

cd "$(dirname "$0")/.."
source venv/bin/activate
streamlit run analysis/runout_explorer.py
