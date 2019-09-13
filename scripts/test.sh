#!/bin/bash

set -ex # fail on first error, print commands

echo "Checking code style with pylint..."
python -m pylint bambi/
echo "Success!"

echo "Running unit tests..."
python3 -m pytest -vx --cov=bambi
echo "Success!"
