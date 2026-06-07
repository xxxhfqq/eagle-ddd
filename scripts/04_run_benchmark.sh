#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results

PYTHONPATH=. python experiments/run_benchmark.py --config configs/local.yaml --method ar --output results/ar.jsonl
PYTHONPATH=. python experiments/run_benchmark.py --config configs/local.yaml --method eagle --output results/eagle.jsonl

PYTHONPATH=. python experiments/run_benchmark.py --config configs/local.yaml --method eagle_ddd --ddd-threshold -3.0 --output results/eagle_ddd_t-3.jsonl
PYTHONPATH=. python experiments/run_benchmark.py --config configs/local.yaml --method eagle_ddd --ddd-threshold -2.0 --output results/eagle_ddd_t-2.jsonl
PYTHONPATH=. python experiments/run_benchmark.py --config configs/local.yaml --method eagle_ddd --ddd-threshold -1.0 --output results/eagle_ddd_t-1.jsonl
PYTHONPATH=. python experiments/run_benchmark.py --config configs/local.yaml --method eagle_ddd --ddd-threshold 0.0 --output results/eagle_ddd_t0.jsonl
