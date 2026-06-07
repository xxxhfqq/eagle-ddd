#!/usr/bin/env bash
set -euo pipefail

mkdir -p third_party

EAGLE_REPO_URL="${EAGLE_REPO_URL:-https://gh-proxy.com/https://github.com/SafeAILab/EAGLE.git}"

if [ ! -d "third_party/EAGLE/.git" ]; then
  git clone --depth=1 "$EAGLE_REPO_URL" third_party/EAGLE
else
  echo "third_party/EAGLE already exists, skip clone."
fi
