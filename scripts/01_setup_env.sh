#!/usr/bin/env bash
set -euo pipefail

python -m pip install numpy pandas matplotlib pyyaml tqdm sentencepiece protobuf accelerate transformers safetensors huggingface-hub
python -m pip install --no-deps -e third_party/EAGLE

echo "Environment dependencies installed."
echo "Note: install PyTorch separately before running benchmarks."
