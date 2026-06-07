#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=. python experiments/smoke_test.py --config configs/local.yaml
