#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR=${STARVLA_DIR:-/root/tianyi/code/starVLA}
LIBERO_PLUS_HOME=${LIBERO_PLUS_HOME:-/root/tianyi/code/LIBERO-plus}
LIBERO_PLUS_VENV=${LIBERO_PLUS_VENV:-${LIBERO_PLUS_HOME}/.venv}
LIBERO_PLUS_REPO=${LIBERO_PLUS_REPO:-https://github.com/sylvestf/LIBERO-plus.git}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
DOWNLOAD_ASSETS=${DOWNLOAD_ASSETS:-true}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

echo "=========================================="
echo " LIBERO-plus uv environment setup"
echo "=========================================="
echo " STARVLA_DIR      : ${STARVLA_DIR}"
echo " LIBERO_PLUS_HOME : ${LIBERO_PLUS_HOME}"
echo " LIBERO_PLUS_VENV : ${LIBERO_PLUS_VENV}"
echo " PYTHON_VERSION   : ${PYTHON_VERSION}"
echo " DOWNLOAD_ASSETS  : ${DOWNLOAD_ASSETS}"
echo "=========================================="

if [[ ! -d "${LIBERO_PLUS_HOME}/.git" ]]; then
    mkdir -p "$(dirname "${LIBERO_PLUS_HOME}")"
    git clone "${LIBERO_PLUS_REPO}" "${LIBERO_PLUS_HOME}"
else
    echo "LIBERO-plus already exists at ${LIBERO_PLUS_HOME}; skipping clone."
fi

if [[ ! -d "${LIBERO_PLUS_VENV}" ]]; then
    uv venv --python "${PYTHON_VERSION}" "${LIBERO_PLUS_VENV}"
else
    echo "Virtualenv already exists at ${LIBERO_PLUS_VENV}; reusing it."
fi

LIBERO_PLUS_PYTHON="${LIBERO_PLUS_VENV}/bin/python"
if [[ ! -x "${LIBERO_PLUS_PYTHON}" ]]; then
    echo "[ERROR] Python not found after uv venv: ${LIBERO_PLUS_PYTHON}" >&2
    exit 1
fi

uv pip install --python "${LIBERO_PLUS_PYTHON}" -U pip setuptools wheel
uv pip install --python "${LIBERO_PLUS_PYTHON}" -r "${STARVLA_DIR}/examples/LIBERO-plus/eval_files/libero_plus_requirements.txt"
uv pip install --python "${LIBERO_PLUS_PYTHON}" numpy==1.24.4 mujoco==3.2.3 tyro matplotlib mediapy websockets msgpack imageio opencv-python-headless scikit-image
uv pip install --python "${LIBERO_PLUS_PYTHON}" -e "${LIBERO_PLUS_HOME}"

"${LIBERO_PLUS_PYTHON}" "${LIBERO_PLUS_VENV}/lib/python3.10/site-packages/robosuite/scripts/setup_macros.py" || true

mkdir -p "${LIBERO_PLUS_HOME}/.libero_config"
cat > "${LIBERO_PLUS_HOME}/.libero_config/config.yaml" <<EOF
assets: ${LIBERO_PLUS_HOME}/libero/libero/assets
bddl_files: ${LIBERO_PLUS_HOME}/libero/libero/bddl_files
benchmark_root: ${LIBERO_PLUS_HOME}/libero/libero
datasets: ${LIBERO_PLUS_HOME}/libero/datasets
init_states: ${LIBERO_PLUS_HOME}/libero/libero/init_files
EOF

if command -v git-lfs >/dev/null 2>&1; then
    git -C "${LIBERO_PLUS_HOME}" lfs install --local || true
    git -C "${LIBERO_PLUS_HOME}" lfs pull || true
fi

if [[ "${DOWNLOAD_ASSETS}" == "true" && ! -e "${LIBERO_PLUS_HOME}/libero/libero/assets" ]]; then
    mkdir -p "${LIBERO_PLUS_HOME}/.hf_assets"
    if [[ ! -f "${LIBERO_PLUS_HOME}/.hf_assets/assets.zip" ]]; then
        curl -L --retry 5 --retry-delay 5 \
            -o "${LIBERO_PLUS_HOME}/.hf_assets/assets.zip" \
            "${HF_ENDPOINT}/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip"
    fi
    unzip -q -o "${LIBERO_PLUS_HOME}/.hf_assets/assets.zip" -d "${LIBERO_PLUS_HOME}/libero/libero"
fi

NESTED_ASSETS="${LIBERO_PLUS_HOME}/libero/libero/inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets"
if [[ ! -e "${LIBERO_PLUS_HOME}/libero/libero/assets" ]] && [[ -d "${NESTED_ASSETS}" ]]; then
    ln -sfn "${NESTED_ASSETS}" "${LIBERO_PLUS_HOME}/libero/libero/assets"
fi

export LIBERO_HOME="${LIBERO_PLUS_HOME}"
export LIBERO_CONFIG_PATH="${LIBERO_PLUS_HOME}/.libero_config"
export PYTHONPATH="${LIBERO_PLUS_HOME}:${PYTHONPATH:-}"

"${LIBERO_PLUS_PYTHON}" - <<'PY'
from libero.libero import benchmark, get_libero_path
print("LIBERO-plus benchmark keys:", sorted(benchmark.get_benchmark_dict().keys()))
print("bddl_files:", get_libero_path("bddl_files"))
print("assets:", get_libero_path("assets"))
PY

if [[ ! -f "${LIBERO_PLUS_HOME}/libero/libero/benchmark/task_classification.json" ]]; then
    echo "[ERROR] Missing task_classification.json under ${LIBERO_PLUS_HOME}." >&2
    exit 1
fi

if [[ ! -d "${LIBERO_PLUS_HOME}/libero/libero/assets" ]]; then
    echo "[WARN] LIBERO-plus assets directory is missing: ${LIBERO_PLUS_HOME}/libero/libero/assets" >&2
    echo "[WARN] Follow the official LIBERO-plus asset instructions, then rerun this script." >&2
fi

echo "LIBERO-plus uv environment ready: ${LIBERO_PLUS_PYTHON}"
