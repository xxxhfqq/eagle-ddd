#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
Automatic source patching is disabled.

This project now uses a manual-edit workflow for DDD integration:
  1. Clone a clean third_party/EAGLE.
  2. Install it with --no-deps.
  3. Apply the documented DDD changes manually to the target files.

Do not run patches/apply_ddd_patch.py directly unless you are debugging it.
See README.md and docs/manual_ddd_changes.md for the manual modification plan.
EOF
