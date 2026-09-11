#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-100}"
FOLD="${2:-0}"
INTERVAL="${INTERVAL:-0.2}"
OUT_DIR="${OUT_DIR:-benchmark_training_seed${SEED}_fold${FOLD}}"
TIME_LOG="${OUT_DIR}/time.txt"
TRAIN_LOG="${OUT_DIR}/train.log"
RESOURCE_LOG="${OUT_DIR}/resources.csv"

mkdir -p "${OUT_DIR}"
rm -f "${TIME_LOG}" "${TRAIN_LOG}" "${RESOURCE_LOG}"
echo "timestamp,pss_kib,gpu_mib" > "${RESOURCE_LOG}"

process_tree() {
    local root="$1"
    local children
    echo "$root"
    children="$(pgrep -P "$root" 2>/dev/null || true)"
    for child in $children; do
        process_tree "$child"
    done
}

echo "Starting RealPhe training benchmark..."
echo "Seed: ${SEED}"
echo "Fold: ${FOLD}"
echo "Polling delay: ${INTERVAL} s"
echo

/usr/bin/time -v -o "${TIME_LOG}" \
    uv run --locked python -m realphe.task.phenotyping.baseline.train \
    solution=baseline seed="${SEED}" fold="${FOLD}" \
    > "${TRAIN_LOG}" 2>&1 &
BENCH_PID=$!

while kill -0 "${BENCH_PID}" 2>/dev/null; do
    mapfile -t PIDS < <(process_tree "${BENCH_PID}" | sort -u)

    PSS_KIB=0
    for pid in "${PIDS[@]}"; do
        if [[ -r "/proc/${pid}/smaps_rollup" ]]; then
            pss="$(awk '/^Pss:/ {print $2}' "/proc/${pid}/smaps_rollup" 2>/dev/null || true)"
            if [[ "${pss}" =~ ^[0-9]+$ ]]; then
                PSS_KIB=$((PSS_KIB + pss))
            fi
        fi
    done

    GPU_MIB=0
    while IFS=',' read -r gpu_pid gpu_mem; do
        gpu_pid="$(echo "${gpu_pid}" | tr -d ' ')"
        gpu_mem="$(echo "${gpu_mem}" | tr -cd '0-9')"
        [[ -z "${gpu_pid}" || -z "${gpu_mem}" ]] && continue
        for pid in "${PIDS[@]}"; do
            if [[ "${gpu_pid}" == "${pid}" ]]; then
                GPU_MIB=$((GPU_MIB + gpu_mem))
                break
            fi
        done
    done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)

    printf '%(%Y-%m-%dT%H:%M:%S)T,%d,%d\n' -1 "${PSS_KIB}" "${GPU_MIB}" >> "${RESOURCE_LOG}"
    sleep "${INTERVAL}"
done

set +e
wait "${BENCH_PID}"
STATUS=$?
set -e

if [[ "${STATUS}" -ne 0 ]]; then
    echo "Training exited with status ${STATUS}. See ${TRAIN_LOG}." >&2
    exit "${STATUS}"
fi

PEAK_PSS_KIB="$(awk -F',' 'NR > 1 && $2 > max {max=$2} END {print max+0}' "${RESOURCE_LOG}")"
PEAK_GPU_MIB="$(awk -F',' 'NR > 1 && $3 > max {max=$3} END {print max+0}' "${RESOURCE_LOG}")"
PEAK_PSS_GIB="$(awk -v x="${PEAK_PSS_KIB}" 'BEGIN {printf "%.3f", x/1024/1024}')"

echo
echo "============================================================"
echo "Training finished"
echo "============================================================"
echo
echo "Training command:"
echo "uv run --locked python -m realphe.task.phenotyping.baseline.train solution=baseline seed=${SEED} fold=${FOLD}"
echo
echo "Hardware:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
echo
echo "Wall-clock / GNU time:"
grep -E 'Elapsed \(wall clock\) time|Maximum resident set size' "${TIME_LOG}" || true
echo
echo "Peak sampled aggregate process-tree PSS:"
echo "  ${PEAK_PSS_KIB} KiB"
echo "  ${PEAK_PSS_GIB} GiB"
echo
echo "Peak sampled GPU memory:"
echo "  ${PEAK_GPU_MIB} MiB"
echo
echo "Logs:"
echo "  Training:  ${TRAIN_LOG}"
echo "  GNU time:  ${TIME_LOG}"
echo "  Resources: ${RESOURCE_LOG}"
