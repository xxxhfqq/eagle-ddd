#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e third_party/EAGLE
python -m pip install -r requirements-extra.txt

echo "Environment dependencies installed."
