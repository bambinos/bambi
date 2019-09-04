#!/bin/bash

set -ex # fail on first error, print commands

echo "Checking code style with pylint..."
python -m pylint bambi/
echo "Success!"
