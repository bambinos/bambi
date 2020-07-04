#!/bin/bash
set -ex # fail on first error, print commands

scripts/lint.sh

echo "Running unit tests..."
python -m pytest -vx --cov=bambi
echo "Success!"
