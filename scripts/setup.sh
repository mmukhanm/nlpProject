#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
nlp_venv="${NLP_VENV:-/workspace/venvs/nlp}"
python3 -c 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 is required"'
python3 -m venv "$nlp_venv"
"$nlp_venv/bin/python" -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r "$project_root/requirements.lock"
"$nlp_venv/bin/python" -m pip install --no-deps -e "$project_root"
cd "$project_root"
"$nlp_venv/bin/python" scripts/download_data.py
export PYTHONHASHSEED=42 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
export MPLCONFIGDIR=/tmp/nlp-matplotlib
"$nlp_venv/bin/python" -m nlp_project.classical
"$nlp_venv/bin/python" -m scripts.build_report
"$nlp_venv/bin/python" -m pytest -q
