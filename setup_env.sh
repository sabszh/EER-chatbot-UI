#!/bin/bash

# Get the directory of the Bash script
scriptDir=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")")

# Create a virtual environment in the same directory as the script
python -m venv "$scriptDir/.venv"

# Activate the virtual environment
source "$scriptDir/.venv/Script/Activate"

# Install requirements from the same directory as the script
python -m pip install -r "$scriptDir/requirements.txt"

echo "Done!"