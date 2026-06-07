#!/usr/bin/env bash
set -euo pipefail

mkdir -p third_party

if [ ! -d "third_party/EAGLE/.git" ]; then
  git clone https://github.com/SafeAILab/EAGLE.git third_party/EAGLE
else
  echo "third_party/EAGLE already exists, skip clone."
fi
