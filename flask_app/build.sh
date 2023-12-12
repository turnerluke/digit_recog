#!/usr/bin/env bash
# exit on error
set -o errexit

pip install --upgrade pip

pip install -r requirements.txt

# Optional: Using poetry
#pip install poetry
#poetry env use 3.11
#
#poetry lock --no-update
#poetry install