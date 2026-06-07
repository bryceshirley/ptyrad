#!/bin/bash
# test_conda_packaging.sh - Sanity check for Conda-Forge local builds

# Exit immediately on error; catch failures inside pipelines too
set -eo pipefail

# Suppress interactive "send error report?" prompts from conda
export CONDA_REPORT_ERRORS=false

LOG_FILE="$HOME/conda_build.log"

## 0. Ensure conda-build is installed
if ! command -v conda-build &> /dev/null; then
    echo "❌ ERROR: conda-build is not installed. Run 'conda install conda-build -y' first."
    exit 1
fi

## 1. Remove old conda build files
echo "🧹 Cleaning old conda build artifacts..."
conda build purge

## 2. Get the expected output path BEFORE building (--output is a fast dry-run)
CONDA_PKG=$(conda build recipe/ --output)
echo "📦 Building the Conda package (output will be: $CONDA_PKG)..."
echo "    (Log: $LOG_FILE — run 'tail -f $LOG_FILE' in another terminal to monitor)"

## 3. Build the conda package
# This will build the package and run the `test` section defined in meta.yaml
conda build recipe/ --no-anaconda-upload 2>&1 | tee "$LOG_FILE"
echo "✅ Package built at: $CONDA_PKG"

## 4. Create a local test environment
# Remove any leftover environment from a previous failed run first
echo "🧪 Creating an isolated virtual environment..."
conda env remove -n test_env_conda --all -y 2>/dev/null || true
conda create -n test_env_conda python=3.12 -y

## 5. Install the built PtyRAD package
# --use-local tells conda to look in your local build cache first
# conda run avoids the conda activate / set -e incompatibility in non-interactive shells
echo "⚙️ Installing the fresh Conda package..."
conda run -n test_env_conda conda install --use-local ptyrad -y

## 6. Test and Verify the 'ptyrad init' behavior
TEST_OUT_DIR="test_init"
echo "🔍 Running integration test script..."

if conda run -n test_env_conda python release/verify_ptyrad_init.py "$TEST_OUT_DIR"; then
    echo "✅ Integration test complete."
else
    echo "❌ Integration test failed."
    conda remove -n test_env_conda --all -y
    exit 1
fi

## 7. Clean up the local test environment
echo "🧹 Cleaning up..."
conda remove -n test_env_conda --all -y

echo "🎉 All tests passed! You are ready to push to conda-forge."
