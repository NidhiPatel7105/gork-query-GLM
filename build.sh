#!/bin/bash
set -e

# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Install requirements
pip install -r requirements.txt

# Verify critical packages
python -c "import setuptools; print('setuptools version:', setuptools.__version__)"
python -c "import fastapi; print('FastAPI installed successfully')"
