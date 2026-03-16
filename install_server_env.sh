#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-environment.yaml}"
ENV_NAME="${ENV_NAME:-wenhai-inference}"
AEROBULK_URL="${AEROBULK_URL:-https://files.pythonhosted.org/packages/3f/8c/7f68eb7ce0508775ffc33bbc3086c652abe4904fb2b5ecf18c200f4cf752/aerobulk-python-0.4.2.tar.gz}"
BUILD_DIR="${BUILD_DIR:-$(pwd)/.build/aerobulk}"

if ! command -v micromamba >/dev/null 2>&1; then
  echo "micromamba is required but not installed."
  exit 1
fi

if ! command -v gfortran >/dev/null 2>&1; then
  echo "gfortran is required to build aerobulk."
  echo "Install it first, for example: sudo apt-get update && sudo apt-get install -y gfortran"
  exit 1
fi

eval "$(micromamba shell hook -s bash)"

if micromamba env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Using existing micromamba environment: ${ENV_NAME}"
else
  micromamba env create -f "${ENV_FILE}"
fi

micromamba activate "${ENV_NAME}"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

if command -v wget >/dev/null 2>&1; then
  wget -q -O aerobulk-python-0.4.2.tar.gz "${AEROBULK_URL}"
elif command -v curl >/dev/null 2>&1; then
  curl -fsSL "${AEROBULK_URL}" -o aerobulk-python-0.4.2.tar.gz
else
  echo "wget or curl is required to download aerobulk."
  exit 1
fi

rm -rf aerobulk-python-0.4.2
tar xzf aerobulk-python-0.4.2.tar.gz
cd aerobulk-python-0.4.2

touch README.md
export SETUPTOOLS_USE_DISTUTILS=stdlib
pip install . --no-build-isolation

python -c "from aerobulk.flux import noskin_np; print('aerobulk OK')"
