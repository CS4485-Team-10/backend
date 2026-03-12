#!/usr/bin/env bash
set -euo pipefail

echo "Setting up git hooks..."
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit

echo "Installing dev dependencies..."
if command -v uv &>/dev/null; then
    uv pip install -r requirements-dev.txt
else
    pip install -r requirements-dev.txt
fi

echo "Done! Pre-commit hooks are now active."
