#!/bin/bash

set -ex # fail on first error, print commands

SRC_DIR=${SRC_DIR:-$(pwd)}

echo "Checking code style with pylint..."
python3 -m pylint bambi/
echo "Success!"
