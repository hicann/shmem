#!/bin/bash

GNPU_NUM=2
USED_CORE=32
WARMUP_LOOPS=50
LOOPS=1000
BYTES_MIN=10
BYTES_MAX=14
UB_SIZE=16

# --- 解析 Bash 脚本接收到的命令行参数 ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --used_core)        USED_CORE="$2"; shift 2 ;;
        --warmup_loops)     WARMUP_LOOPS="$2"; shift 2 ;;
        --loops)            LOOPS="$2"; shift 2 ;;
        --bytes_in_exp_min) BYTES_MIN="$2"; shift 2 ;;
        --bytes_in_exp_max) BYTES_MAX="$2"; shift 2 ;;
        --ub_size)          UB_SIZE="$2"; shift 2 ;;
        --help)
            cat <<'EOF'
Usage: run.sh [options]

Launches a fixed 2-card (Active PE0 / Passive PE1) gm2gm RMA perf test.

Options:
  --used_core <int>         Cores (blocks) used per PE.                 (default: 32)
  --warmup_loops <int>      Warmup iterations, not counted in stats.    (default: 50)
  --loops <int>             Sampled iterations, averaged for results.   (default: 1000)
  --bytes_in_exp_min <int>  Lower transfer-size exponent; size = 2^min bytes. (default: 10)
  --bytes_in_exp_max <int>  Upper transfer-size exponent; size = 2^max bytes. (default: 14)
  --ub_size <int>           Unified Buffer size per core in KB.         (default: 16)
                            Only affects the SIMD path; ignored in SIMT (default) mode.
  --help                    Show this message.

The test sweeps 2^min .. 2^max bytes, one CSV row per size.
EOF
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# --- 环境与路径设置 ---
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $(dirname "$SCRIPT_DIR"))

EXAMPLE=simt_rma_perftest
EXECUTABLE="${PROJECT_ROOT}/build/bin/${EXAMPLE}"

export SHMEM_UID_SESSION_ID=127.0.0.1:8899
export ACLSHMEM_UID_SESSION_ID=127.0.0.1:8899
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH

cd "${SCRIPT_DIR}"

pids=()
cleanup() {
    echo -e "\n[Terminating] Caught Ctrl+C, killing background processes..."
    if [ ${#pids[@]} -ne 0 ]; then
        kill "${pids[@]}" 2>/dev/null
    fi
    exit 1
}
trap cleanup SIGINT SIGTERM
echo "Starting $GNPU_NUM processes..."

for (( idx = 0; idx < ${GNPU_NUM}; idx = idx + 1 )); do
    "${EXECUTABLE}" \
        --mype "${idx}" \
        --used_core "${USED_CORE}" \
        --warmup_loops "${WARMUP_LOOPS}" \
        --loops "${LOOPS}" \
        --bytes_in_exp_min "${BYTES_MIN}" \
        --bytes_in_exp_max "${BYTES_MAX}" \
        --ub_size "${UB_SIZE}" &
    
    pid=$!
    pids+=("$pid")
    echo "[Rank $idx] Started with PID $pid"
done

ret=0
for pid in "${pids[@]}"; do
    wait "$pid"
    cur_ret=$?
    if [[ $cur_ret -ne 0 ]]; then
        ret=$cur_ret
    fi
done

echo "All processes done. Exit code: $ret"
cd "${CURRENT_DIR}"
exit $ret