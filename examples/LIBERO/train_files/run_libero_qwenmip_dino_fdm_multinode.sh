#!/usr/bin/env bash
# Multi-node launcher for QwenMIPDINOFDM LIBERO training.
#
# Submit/run this same command as the entrypoint of every node:
#
#   NNODES=2 bash examples/LIBERO/train_files/run_libero_qwenmip_dino_fdm_multinode.sh
#
# The wrapper tries to infer NODE_RANK and MASTER_ADDR from common cluster
# environment variables or from a Kubernetes StatefulSet hostname. They can
# always be provided explicitly by the job launcher.
set -euo pipefail

STARVLA_ROOT=${STARVLA_ROOT:-/root/tianyi/code/starVLA}
SINGLE_NODE_SCRIPT="${STARVLA_ROOT}/examples/LIBERO/train_files/run_libero_qwenmip_dino_fdm.sh"

if [[ ! -f "${SINGLE_NODE_SCRIPT}" ]]; then
  echo "FATAL: cannot find ${SINGLE_NODE_SCRIPT}; set STARVLA_ROOT correctly." >&2
  exit 1
fi

# Number of nodes. Prefer an explicit NNODES; WORLD_SIZE is used by some
# platforms for the number of worker nodes.
NNODES=${NNODES:-${WORLD_SIZE:-}}
if [[ -z "${NNODES}" ]]; then
  echo "FATAL: NNODES is not set and WORLD_SIZE is unavailable." >&2
  echo "       Set NNODES to the number of nodes, e.g. NNODES=2." >&2
  exit 1
fi

NPROC_PER_NODE=${NPROC_PER_NODE:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}}
[[ "${NPROC_PER_NODE}" == "auto" ]] && NPROC_PER_NODE=8
MASTER_PORT=${MASTER_PORT:-29500}

# Rank supplied by the scheduler/platform takes precedence.
if [[ -z "${NODE_RANK:-}" ]]; then
  if [[ -n "${VC_TASK_INDEX:-}" ]]; then
    NODE_RANK=${VC_TASK_INDEX}
  elif [[ -n "${RANK:-}" ]]; then
    NODE_RANK=${RANK}
  elif [[ -n "${HOST_INDEX:-}" ]]; then
    NODE_RANK=${HOST_INDEX}
  else
    FQDN=$(hostname -f)
    SHORT=${FQDN%%.*}
    CANDIDATE=${SHORT##*-}
    if [[ "${CANDIDATE}" =~ ^[0-9]+$ ]]; then
      NODE_RANK=${CANDIDATE}
    else
      echo "FATAL: cannot infer NODE_RANK from hostname '${FQDN}'." >&2
      echo "       Set NODE_RANK explicitly for this job." >&2
      exit 1
    fi
  fi
fi

# MASTER_ADDR may be injected by Slurm/Kubernetes/the job launcher. Otherwise
# derive the rank-0 StatefulSet hostname: <job>-0.<service>.<namespace>...
if [[ -z "${MASTER_ADDR:-}" ]]; then
  FQDN=${FQDN:-$(hostname -f)}
  SHORT=${FQDN%%.*}
  DOMAIN=${FQDN#*.}
  JOB_PREFIX=${SHORT%-*}

  if [[ "${DOMAIN}" != "${FQDN}" && "${SHORT##*-}" =~ ^[0-9]+$ ]]; then
    MASTER_ADDR="${JOB_PREFIX}-0.${DOMAIN}"
  else
    echo "FATAL: cannot infer MASTER_ADDR from hostname '${FQDN}'." >&2
    echo "       Set MASTER_ADDR to the reachable address of node 0." >&2
    exit 1
  fi
fi

if (( NNODES < 2 )); then
  echo "WARNING: NNODES=${NNODES}; use the single-node script directly for one node." >&2
fi

if (( NODE_RANK < 0 || NODE_RANK >= NNODES )); then
  echo "FATAL: NODE_RANK=${NODE_RANK} is outside [0, $((NNODES - 1))]." >&2
  exit 1
fi

export NNODES NPROC_PER_NODE NODE_RANK MASTER_ADDR MASTER_PORT

# Cross-node NCCL defaults. Override these when the cluster uses a different
# network interface or has a dedicated InfiniBand configuration.
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
if [[ -d /dev/infiniband ]]; then
  export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
else
  export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
fi
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-3600}
export NCCL_SOCKET_TIMEOUT_MS=${NCCL_SOCKET_TIMEOUT_MS:-360000}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-0}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}

# Avoid multiple nodes appending to the same shell log on a shared filesystem.
OUTPUT_DIR=${run_root_dir:-/root/tianyi/starVLA/playground/Checkpoints/libero}/${run_id:-libero_qwenmip_dino_fdm_state7_100k}
export TRAIN_LOG=${TRAIN_LOG:-${OUTPUT_DIR}/train.node${NODE_RANK}.log}

echo "=== QwenMIPDINOFDM LIBERO multi-node ==="
echo "hostname       : $(hostname -f)"
echo "NNODES         : ${NNODES}"
echo "NODE_RANK      : ${NODE_RANK}"
echo "NPROC_PER_NODE : ${NPROC_PER_NODE}"
echo "TOTAL_GPUS     : $((NNODES * NPROC_PER_NODE))"
echo "MASTER         : ${MASTER_ADDR}:${MASTER_PORT}"
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
echo "TRAIN_LOG      : ${TRAIN_LOG}"

# Wait briefly for rank 0 DNS/service discovery before starting Accelerate.
if command -v getent >/dev/null 2>&1; then
  echo "Resolving ${MASTER_ADDR} ..."
  for attempt in $(seq 1 60); do
    if getent hosts "${MASTER_ADDR}" >/dev/null 2>&1; then
      echo "  -> $(getent hosts "${MASTER_ADDR}" | head -n 1)"
      break
    fi
    if (( attempt == 60 )); then
      echo "FATAL: ${MASTER_ADDR} did not resolve after 5 minutes." >&2
      exit 1
    fi
    sleep 5
  done
fi

exec bash "${SINGLE_NODE_SCRIPT}" "$@"
