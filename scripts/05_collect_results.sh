#!/usr/bin/env bash
python experiments/summarize_results.py --results-dir results --output results/summary.csv
python experiments/plot_results.py --summary results/summary.csv --depths results/ddd_depths.csv --output-dir results
