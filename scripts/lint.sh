#!/bin/bash
set -ex # fail on first error, print commands

echo "Checking code style with black...."
python -m black bambi --check
echo "Success!"

echo "Checking code style with pylint..."
python -m pylint bambi/
echo "Success!"
