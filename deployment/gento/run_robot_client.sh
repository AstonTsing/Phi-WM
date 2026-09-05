#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${GENTO_CONFIG:-${SCRIPT_DIR}/config/gento_insert_usb.yaml}"

if [[ -n "${GENTO_ROS_SETUP:-}" ]]; then
    if [[ ! -f "${GENTO_ROS_SETUP}" ]]; then
        echo "GENTO_ROS_SETUP not found: ${GENTO_ROS_SETUP}" >&2
        exit 2
    fi
    set +u
    # shellcheck source=/dev/null
    source "${GENTO_ROS_SETUP}"
    set -u
fi
if [[ -n "${GENTO_MSGS_SETUP:-}" ]]; then
    if [[ ! -f "${GENTO_MSGS_SETUP}" ]]; then
        echo "GENTO_MSGS_SETUP not found: ${GENTO_MSGS_SETUP}" >&2
        exit 2
    fi
    set +u
    # shellcheck source=/dev/null
    source "${GENTO_MSGS_SETUP}"
    set -u
fi

GENTO_PYTHON="${GENTO_PYTHON:-python3}"
if [[ ! -f "${CONFIG}" ]]; then
    echo "Gento config not found: ${CONFIG}" >&2
    exit 2
fi

export PYTHONPATH="${STARVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${GENTO_PYTHON}" -c \
    'import cv2, msgpack, numpy, rclpy, websockets, yaml; import marvin_msgs.msg; import sensor_msgs.msg; import deployment.gento.eval'

cd "${STARVLA_ROOT}"
exec "${GENTO_PYTHON}" -m deployment.gento.eval -c "${CONFIG}" "$@"
