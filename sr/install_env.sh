#!/bin/bash
# Create the `sr` conda env and trigger PySR's Julia install (Task 3 Step 1).
set -e
C=${CONDA_ROOT:-$HOME/miniforge3}
$C/bin/conda create -y -n sr python=3.11 numpy pandas scipy sympy pytest pyarrow
$C/envs/sr/bin/pip install pysr
$C/envs/sr/bin/python -c "import pysr, sys; print('pysr', pysr.__version__); from pysr import jl; print(jl.seval('string(VERSION)')); print(jl.seval('using SymbolicRegression; string(pkgversion(SymbolicRegression))'))"
echo INSTALL_DONE
