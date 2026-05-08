#!/bin/bash

# Define the path for the virtual environment
BASE_DIR=~/envs
VENV_DIR="tsdiscovery-variablelags-env"
VENV_PATH="$BASE_DIR/$VENV_DIR"

mkdir -p "$BASE_DIR"

if [ -d "$VENV_PATH" ]; then
    echo "Virtual environment exists, removing it..."
    rm -rf "$VENV_PATH"
fi

echo "Creating new virtual environment at $VENV_PATH ..."
python3 -m venv "$VENV_PATH"

echo "Activating virtual environment..."
source  "$VENV_PATH/bin/activate"

echo "Upgrading pip and installing packages..."
python -m pip install -U pip

python -m pip install -e .

echo "Activating jupyter kernel..."
python -m ipykernel install --user --name market-exit --display-name "market-exit"

echo "Virtual environment is ready and activated!"
