#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Sweep relay vs v3 barrier latency on server215 (910B).
# Usage:
#   source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
#   bash scripts/build.sh -uttests
#   bash scripts/benchmark_relay_barrier_perf.sh
# Optional env:
#   PE_LIST="2 4 6 8 16"  GNPU=8  FIRST_NPU=0  PERF_ITERS=50
# -----------------------------------------------------------------------------------------------------------
set -euo pipefail

readonly SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
readonly PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
readonly BUILD_PATH="${PROJECT_ROOT}/build"
readonly OUT_DIR="${PROJECT_ROOT}/docs/reports/relay_barrier_perf_runs"
readonly PE_LIST="${PE_LIST:-2 4 6 8 16}"
readonly GNPU="${GNPU:-8}"
readonly FIRST_NPU="${FIRST_NPU:-0}"
readonly IPPORT="${IPPORT:-tcp://127.0.0.1:8666}"
readonly TEST_FILTER="${TEST_FILTER:-test_barrier_relay_perf}"

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    if [ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]; then
        # shellcheck source=/dev/null
        source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
    fi
fi

export SMEM_CONF_STORE_TLS_ENABLE=0
export LD_LIBRARY_PATH="${PROJECT_ROOT}/build/lib:${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/lib64:${LD_LIBRARY_PATH:-}"

killall -9 aclshmem_unittest 2>/dev/null || true
mkdir -p "${OUT_DIR}"
STAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="${OUT_DIR}/summary_${STAMP}.csv"
echo "npes,v3_ms,relay_ms,put_relay_ms,ratio,status" > "${SUMMARY}"

cd "${BUILD_PATH}"
for npes in ${PE_LIST}; do
    if [ "${npes}" -gt "${GNPU}" ] && [ $((npes % GNPU)) -ne 0 ]; then
        echo "[WARN] npes=${npes} with gnpu=${GNPU}: ranks map round-robin to devices"
    fi
    LOG="${OUT_DIR}/npes${npes}_${STAMP}.log"
    echo "=== npes=${npes} gnpu=${GNPU} ===" | tee "${LOG}"
    set +e
    ./bin/aclshmem_unittest "${npes}" "${IPPORT}" "${GNPU}" 0 "${FIRST_NPU}" \
        --gtest_filter="*${TEST_FILTER}*" 2>&1 | tee -a "${LOG}"
    RC=${PIPESTATUS[0]}
    set -e
    if [ "${RC}" -ne 0 ]; then
        echo "${npes},,,,,FAIL" >> "${SUMMARY}"
        echo "[FAIL] npes=${npes} rc=${RC}" | tee -a "${LOG}"
        continue
    fi
    CSV_LINE=$(grep '\[PERF_CSV\]' "${LOG}" | tail -1 || true)
    if [ -z "${CSV_LINE}" ]; then
        echo "${npes},,,,,NO_CSV" >> "${SUMMARY}"
        continue
    fi
    # [PERF_CSV] npes=8,v3_ms=0.036,relay_ms=0.016,put_relay_ms=0.042,ratio=0.44
    V3=$(echo "${CSV_LINE}" | sed -n 's/.*v3_ms=\([^,]*\).*/\1/p')
    RELAY=$(echo "${CSV_LINE}" | sed -n 's/.*,relay_ms=\([^,]*\).*/\1/p')
    PUT=$(echo "${CSV_LINE}" | sed -n 's/.*put_relay_ms=\([^,]*\).*/\1/p')
    RATIO=$(echo "${CSV_LINE}" | sed -n 's/.*,ratio=\([^ ]*\).*/\1/p')
    echo "${npes},${V3},${RELAY},${PUT},${RATIO},PASS" >> "${SUMMARY}"
    echo "[PASS] npes=${npes} relay/v3=${RATIO}" | tee -a "${LOG}"
done

echo "Summary: ${SUMMARY}"
cat "${SUMMARY}"
