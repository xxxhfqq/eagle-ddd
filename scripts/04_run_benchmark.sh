#!/usr/bin/env bash
mkdir -p results

python experiments/run_benchmark.py --config configs/local.yaml --method ar --output results/ar.jsonl
python experiments/run_benchmark.py --config configs/local.yaml --method eagle --output results/eagle.jsonl

python experiments/run_benchmark.py --config configs/local.yaml --method eagle_ddd --ddd-threshold -3.0 --output results/eagle_ddd_t-3.jsonl
python experiments/run_benchmark.py --config configs/local.yaml --method eagle_ddd --ddd-threshold -2.0 --output results/eagle_ddd_t-2.jsonl
python experiments/run_benchmark.py --config configs/local.yaml --method eagle_ddd --ddd-threshold -1.0 --output results/eagle_ddd_t-1.jsonl
python experiments/run_benchmark.py --config configs/local.yaml --method eagle_ddd --ddd-threshold 0.0 --output results/eagle_ddd_t0.jsonl
