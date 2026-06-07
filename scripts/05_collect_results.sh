#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python experiments/summarize_results.py --results-dir results --output results/summary.csv
python experiments/plot_results.py --summary results/summary.csv --depths results/ddd_depths.csv --output-dir results
