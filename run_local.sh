#!/usr/bin/env bash
# No Docker, no database: uses CSV exports. Needs Python 3.10-3.12 and Java 17+ installed.
set -euo pipefail
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q -r requirements.txt
python -m kartmaint.generate_data --sink csv
python -m kartmaint.etl --source csv
python -m kartmaint.features
python -m kartmaint.train
python -m kartmaint.score
echo "Done. Start the dashboard with:  streamlit run dashboard/app.py"
