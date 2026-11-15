#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DATA_DIR="${DATA_DIR:-${ROOT}/reproduce/data}"
SAVE_DIR="${SAVE_DIR:-${ROOT}/save_reproduce}"
LOG_DIR="${LOG_DIR:-${SAVE_DIR}/_logs_tsa_all}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS_CSV="${TSA_GPUS:-0}"
SEEDS_CSV="${TSA_SEEDS:-40,41,42}"
CFGS_CSV="${TSA_CFGS:-config-tsa/tsa-movi_c.py,config-tsa/tsa-movi_e.py,config-tsa/tsa-ytvis_hq.py,config-tsa/tsa-ovis.py}"
SKIP_IF_DONE="${SKIP_IF_DONE:-1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
FAIL_FAST="${FAIL_FAST:-1}"

mkdir -p "${SAVE_DIR}" "${LOG_DIR}"

IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
IFS=',' read -r -a CFGS <<< "${CFGS_CSV}"

for d in ytvis_hq movi_c movi_e ovis; do
  if [[ ! -e "${DATA_DIR}/${d}/train.lmdb" ]]; then
    echo "ERROR: missing ${DATA_DIR}/${d}/train.lmdb" >&2
    exit 1
  fi
done

JOBS=()
for cfg in "${CFGS[@]}"; do
  cfg="$(echo "${cfg}" | xargs)"
  [[ -z "${cfg}" ]] && continue
  if [[ ! -f "${cfg}" ]]; then
    echo "ERROR: config not found: ${cfg}" >&2
    exit 1
  fi
  for seed in "${SEEDS[@]}"; do
    seed="$(echo "${seed}" | xargs)"
    [[ -z "${seed}" ]] && continue
    JOBS+=("${cfg}|${seed}")
  done
done

echo "=== TSA full train ==="
echo "data_dir=${DATA_DIR}"
echo "save_dir=${SAVE_DIR}"
echo "gpus=${GPUS[*]}  jobs_per_gpu=${JOBS_PER_GPU}  max_parallel=$(( ${#GPUS[@]} * JOBS_PER_GPU ))"
echo "seeds=${SEEDS[*]}"
echo "jobs=${#JOBS[@]} (configs × seeds)"
echo "python=$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
"${PYTHON_BIN}" - <<'PY'
import torch
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} ndev={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available inside training environment")
PY

is_done() {
  local cfg="$1"
  local seed="$2"
  local name
  name="$(basename "${cfg}" .py)"
  local done_mark="${SAVE_DIR}/${name}/${seed}/.done"
  local log_txt="${SAVE_DIR}/${name}/${seed}.txt"
  if [[ "${SKIP_IF_DONE}" != "1" ]]; then
    return 1
  fi
  if [[ -f "${done_mark}" ]]; then
    return 0
  fi
  if [[ -f "${log_txt}" ]]; then
    if "${PYTHON_BIN}" - "$cfg" "$log_txt" <<'PY'
import json, re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text()
m_step = re.search(r"^total_step\s*=\s*(\d+)", text, re.M)
m_div = re.search(r"^val_interval\s*=\s*total_step\s*//\s*(\d+)", text, re.M)
if not m_step:
    raise SystemExit(1)
total_step = int(m_step.group(1))
div = int(m_div.group(1)) if m_div else 40
val_interval = max(1, total_step // div)
need = max(1, total_step // val_interval)
n_val = 0
for line in Path(sys.argv[2]).read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    keys = json.loads(line)
    n_val += sum(1 for k in keys if str(k).endswith("-val"))
raise SystemExit(0 if n_val >= need else 1)
PY
    then
      mkdir -p "${SAVE_DIR}/${name}/${seed}"
      date -u +"%Y-%m-%dT%H:%M:%SZ recovered" > "${done_mark}"
      return 0
    fi
  fi
  return 1
}

run_one() {
  local gpu="$1"
  local cfg="$2"
  local seed="$3"
  local name
  name="$(basename "${cfg}" .py)"
  local out_dir="${SAVE_DIR}/${name}/${seed}"
  local log="${LOG_DIR}/${name}-seed${seed}-gpu${gpu}.log"

  if is_done "${cfg}" "${seed}"; then
    echo "[skip] ${name} seed=${seed} (already done in ${out_dir})"
    return 0
  fi

  echo "[start] gpu=${gpu} ${name} seed=${seed} -> ${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" train.py \
      --seed "${seed}" \
      --cfg_file "${cfg}" \
      --data_dir "${DATA_DIR}" \
      --save_dir "${SAVE_DIR}" \
      > "${log}" 2>&1
  local rc=$?
  if [[ ${rc} -ne 0 ]]; then
    echo "[fail] gpu=${gpu} ${name} seed=${seed} rc=${rc} (see ${log})" >&2
    return ${rc}
  fi
  mkdir -p "${out_dir}"
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "${out_dir}/.done"
  echo "[done] gpu=${gpu} ${name} seed=${seed}"
}

declare -A SLOT_PID=()
fail=0

reap_finished() {
  local key
  for key in "${!SLOT_PID[@]}"; do
    local pid="${SLOT_PID[$key]}"
    if ! kill -0 "${pid}" 2>/dev/null; then
      local rc=0
      wait "${pid}" || rc=$?
      unset "SLOT_PID[$key]"
      if [[ ${rc} -ne 0 ]]; then
        echo "[fail] slot=${key} worker rc=${rc}" >&2
        fail=1
      fi
    fi
  done
}

pick_free_slot() {
  local gpu slot
  for gpu in "${GPUS[@]}"; do
    for ((slot = 0; slot < JOBS_PER_GPU; slot++)); do
      local key="${gpu}#${slot}"
      if [[ -z "${SLOT_PID[$key]:-}" ]]; then
        echo "${key}"
        return 0
      fi
    done
  done
  return 1
}

for job in "${JOBS[@]}"; do
  if [[ "${FAIL_FAST}" == "1" && "${fail}" -ne 0 ]]; then
    echo "[abort] FAIL_FAST=1 after earlier failure; remaining jobs skipped" >&2
    break
  fi
  cfg="${job%%|*}"
  seed="${job##*|}"
  local_slot=""
  while true; do
    reap_finished
    if [[ "${FAIL_FAST}" == "1" && "${fail}" -ne 0 ]]; then
      break 2
    fi
    if local_slot="$(pick_free_slot)"; then
      break
    fi
    sleep 5
  done
  gpu="${local_slot%%#*}"
  (
    run_one "${gpu}" "${cfg}" "${seed}"
  ) &
  SLOT_PID[$local_slot]=$!
done

while (( ${#SLOT_PID[@]} > 0 )); do
  reap_finished
  sleep 2
done

if [[ ${fail} -ne 0 ]]; then
  echo "=== TSA train finished with failures; check ${LOG_DIR} ===" >&2
  exit 1
fi
echo "=== TSA train finished OK (${#JOBS[@]} jobs) ==="
