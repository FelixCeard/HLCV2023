#!/usr/bin/env bash

# Path to the `.py` file you want to run
#PYTHON_SCRIPT_PATH="/home/hlcv_team017/HLCV2023/"
# Path to the Python binary of the conda environment
#CONDA_PYTHON_BINARY_PATH="/home/hlcv_team017/miniconda3/envs/hlcv-ss23/bin/python"

#cd $PYTHON_SCRIPT_PATH
#$CONDA_PYTHON_BINARY_PATH "$@"

python -m pip install -r requirements.txt

python "$@"