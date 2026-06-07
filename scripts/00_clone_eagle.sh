#!/usr/bin/env bash
set -euo pipefail

mkdir -p third_party

TARGET_DIR="third_party/EAGLE"
DEFAULT_URL="https://github.com/SafeAILab/EAGLE.git"
REPO_URL="${EAGLE_REPO_URL:-$DEFAULT_URL}"
BRANCH="${EAGLE_BRANCH:-main}"
DEPTH="${EAGLE_CLONE_DEPTH:-1}"

if [ -d "$TARGET_DIR/.git" ]; then
  echo "$TARGET_DIR already exists, skip clone."
  exit 0
fi

echo "Cloning EAGLE from: $REPO_URL"
echo "Branch: $BRANCH | Depth: $DEPTH"

git clone --depth "$DEPTH" --branch "$BRANCH" "$REPO_URL" "$TARGET_DIR"
