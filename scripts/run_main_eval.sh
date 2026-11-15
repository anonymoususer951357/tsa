#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

GPUS_CSV="${TSA_GPUS:-0}"
IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${ROOT}/reproduce/_logs_main_eval"
OUT_JSON="${ROOT}/reproduce/main_table_results.txt"
RESUME="${RESUME:-0}"

mkdir -p "${LOG_DIR}"

if [[ "${RESUME}" != "1" ]]; then
  rm -f "${OUT_JSON}"
  echo "fresh run: wiped ${OUT_JSON}"
else
  echo "RESUME=1: keep existing results in ${OUT_JSON}"
fi

"${PYTHON_BIN}" reproduce/eval_main_tables.py --list_only

mapfile -t JOBS < <("${PYTHON_BIN}" - <<'PY'
from reproduce.eval_main_tables import all_jobs, job_id, read_done, OUT_JSON
done = read_done(OUT_JSON)
for j in all_jobs():
    if job_id(j) in done:
        continue
    print(f"{j['method']}|{j['dataset']}|{j['seed']}|{j['cfg_file']}|{j['ckpt_file']}|{j['patch_tag']}")
PY
)

echo "pending jobs: ${#JOBS[@]}"
if (( ${#JOBS[@]} == 0 )); then
  "${PYTHON_BIN}" reproduce/eval_main_tables.py --summary_only
  exit 0
fi

declare -A GPU_PID=()
fail=0

reap() {
  local gpu
  for gpu in "${!GPU_PID[@]}"; do
    local pid="${GPU_PID[$gpu]}"
    if ! kill -0 "${pid}" 2>/dev/null; then
      local rc=0
      wait "${pid}" || rc=$?
      unset "GPU_PID[$gpu]"
      if [[ ${rc} -ne 0 ]]; then
        echo "[fail] gpu=${gpu} rc=${rc}" >&2
        fail=1
      fi
    fi
  done
}

pick() {
  local gpu
  for gpu in "${GPUS[@]}"; do
    if [[ -z "${GPU_PID[$gpu]:-}" ]]; then
      echo "${gpu}"
      return 0
    fi
  done
  return 1
}

run_job() {
  local gpu="$1"
  local spec="$2"
  IFS='|' read -r method dataset seed cfg ckpt patch <<< "${spec}"
  local log="${LOG_DIR}/${method}-${dataset}-seed${seed}-gpu${gpu}.log"
  echo "[start] gpu=${gpu} ${method} ${dataset} seed=${seed} -> ${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" - <<PY > "${log}" 2>&1
import json, os, sys
from pathlib import Path
import numpy as np
import torch as pt

sys.path.insert(0, "${ROOT}")
os.chdir("${ROOT}")
from eval_archives import evaluate, job_id, set_seed
from reproduce.eval_main_tables import OUT_JSON, _resolve_patch

tags = [] if "${patch}" == "none" else [t for t in "${patch}".split(",") if t]
job = dict(
    method="${method}",
    dataset="${dataset}",
    seed=int("${seed}"),
    cfg_file=Path("${cfg}"),
    ckpt_file=Path("${ckpt}"),
    patch=_resolve_patch(tags),
)
set_seed(job["seed"])
pt.backends.cudnn.benchmark = True
data_dir = Path(os.environ.get("DATA_DIR", "${ROOT}/reproduce/data"))
info = evaluate(job, data_dir)
rec = dict(
    id=job_id(job),
    method=job["method"],
    dataset=job["dataset"],
    seed=job["seed"],
    ckpt=str(job["ckpt_file"]),
    cfg=str(job["cfg_file"]),
    metrics={k: float(v) for k, v in info.items() if np.isscalar(v)},
)
with open(OUT_JSON, "a") as f:
    f.write("RESULT " + json.dumps(rec) + "\n")
print("OK", rec["id"])
print(json.dumps({k: rec["metrics"][k] for k in ("ari_fg", "mbo", "hota", "idf1") if k in rec["metrics"]}, indent=2))
PY
}

for spec in "${JOBS[@]}"; do
  if [[ "${fail}" -ne 0 ]]; then
    echo "[abort] earlier failure" >&2
    break
  fi
  while true; do
    reap
    if gpu="$(pick)"; then
      break
    fi
    sleep 5
  done
  ( run_job "${gpu}" "${spec}" ) &
  GPU_PID[$gpu]=$!
done

while (( ${#GPU_PID[@]} > 0 )); do
  reap
  sleep 2
done

if [[ ${fail} -ne 0 ]]; then
  echo "eval finished with failures; see ${LOG_DIR}" >&2
  exit 1
fi

"${PYTHON_BIN}" reproduce/eval_main_tables.py --summary_only
echo "done -> reproduce/main_table_results.txt"
