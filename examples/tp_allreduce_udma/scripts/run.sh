#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
example_dir="$(cd "${script_dir}/.." && pwd)"
project_root="$(cd "${example_dir}/../.." && pwd)"

die() {
    echo "Error: $*" >&2
    exit 1
}

is_uint() {
    [[ "$1" =~ ^(0|[1-9][0-9]*)$ ]]
}

mode="baseline"
pes=""
first_npu=0
elements=1024
iterations=1
warmup_count=""
aiv_count=""
port=8899
dtype="float16_t"
tp_size=2
perf_mode=0
prof_pe=0
perf_csv=""
tailcut_ratio="2:1:1"
ratio_specified=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode|-pes|-fnpu|-elements|-type|-aiv|--loops|--warmup|-port|--prof-pe|--perf-csv|--ratio)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            ;;
    esac
    case "$1" in
        --mode) mode="$2"; shift 2 ;;
        -pes) pes="$2"; shift 2 ;;
        -fnpu) first_npu="$2"; shift 2 ;;
        -elements) elements="$2"; shift 2 ;;
        -type) dtype="$2"; shift 2 ;;
        -aiv) aiv_count="$2"; shift 2 ;;
        --loops) iterations="$2"; shift 2 ;;
        --warmup) warmup_count="$2"; shift 2 ;;
        -port) port="$2"; shift 2 ;;
        --perf) perf_mode=1; shift ;;
        --prof-pe) prof_pe="$2"; shift 2 ;;
        --perf-csv) perf_csv="$2"; shift 2 ;;
        --ratio) tailcut_ratio="$2"; ratio_specified=1; shift 2 ;;
        *) die "unknown option $1" ;;
    esac
done

[[ "${mode}" == "baseline" || "${mode}" == "tailcut" ]] || die "mode must be baseline or tailcut, got ${mode}"

if [[ "${mode}" == "tailcut" ]]; then
    pes="${pes:-4}"
    aiv_count="${aiv_count:-3}"
else
    pes="${pes:-2}"
    aiv_count="${aiv_count:-1}"
fi
warmup_count="${warmup_count:-${perf_mode}}"
perf_csv="${perf_csv:-output/${mode}_perf.csv}"

is_uint "${pes}" || die "pes must be a non-negative integer"
is_uint "${first_npu}" || die "first_npu must be a non-negative integer"
is_uint "${elements}" || die "elements must be a non-negative integer"
is_uint "${iterations}" || die "loops must be a non-negative integer"
is_uint "${warmup_count}" || die "warmup must be a non-negative integer"
is_uint "${aiv_count}" || die "aiv_count must be a non-negative integer"
is_uint "${port}" || die "port must be a non-negative integer"
is_uint "${prof_pe}" || die "prof_pe must be a non-negative integer"

[[ "${dtype}" == "int32_t" || "${dtype}" == "int" || "${dtype}" == "float16_t" ]] ||
    die "dtype must be int32_t or float16_t, got ${dtype}"
(( elements > 0 && elements % tp_size == 0 )) ||
    die "elements must be positive and divisible by ${tp_size}"
(( iterations > 0 )) || die "loops must be positive"
(( port > 0 && port <= 65535 )) || die "port must be in [1, 65535]"

if (( perf_mode == 0 )); then
    (( iterations == 1 )) || die "--loops requires --perf; functional mode runs one measured iteration"
    (( warmup_count == 0 )) || die "nonzero --warmup requires --perf"
fi

if [[ "${mode}" == "baseline" ]]; then
    [[ ${pes} -eq 2 || ${pes} -eq 4 || ${pes} -eq 8 || ${pes} -eq 16 ]] ||
        die "baseline pes must be 2, 4, 8, or 16"
    (( aiv_count > 0 )) || die "baseline aiv_count must be positive"
    (( ratio_specified == 0 )) || die "--ratio is valid only in tailcut mode"
    tailcut_ratio="1:0:0"
else
    [[ ${pes} -eq 4 || ${pes} -eq 8 || ${pes} -eq 16 ]] || die "tailcut pes must be 4, 8, or 16"
    (( aiv_count >= 3 )) || die "tailcut aiv_count must be at least 3 for concurrent path submission"
    if [[ ! "${tailcut_ratio}" =~ ^(0|[1-9][0-9]*):(0|[1-9][0-9]*):(0|[1-9][0-9]*)$ ]]; then
        die "ratio must be direct:relay0:relay1 with non-negative integer weights, got ${tailcut_ratio}"
    fi
    direct_weight="${BASH_REMATCH[1]}"
    relay0_weight="${BASH_REMATCH[2]}"
    relay1_weight="${BASH_REMATCH[3]}"
    [[ "${direct_weight}" != "0" || "${relay0_weight}" != "0" || "${relay1_weight}" != "0" ]] ||
        die "ratio weight sum must be positive"
    tailcut_ratio="${direct_weight}:${relay0_weight}:${relay1_weight}"
fi

if (( perf_mode != 0 && prof_pe >= pes )); then
    die "prof_pe must be in [0, pes)"
fi

binary=${project_root}/build/bin/tp_allreduce_udma
if [[ ! -x ${binary} ]]; then
    if [[ "${mode}" == "tailcut" ]]; then
        die "missing ${binary}; build with: bash scripts/build.sh -examples -soc_type Ascend950 -enable_relay"
    fi
    die "missing ${binary}; build with: bash scripts/build.sh -examples -soc_type Ascend950"
fi

cd "${example_dir}"
mkdir -p golden output
for ((pe_id = 0; pe_id < pes; ++pe_id)); do
    rm -f -- "output/output_${pe_id}.bin"
done
if (( perf_mode != 0 )); then
    mkdir -p "$(dirname "${perf_csv}")"
fi

python3 "${script_dir}/gen_data.py" \
    --pes "${pes}" \
    --elements "${elements}" \
    --tp-size "${tp_size}" \
    --dtype "${dtype}"

export PROJECT_ROOT=${project_root}
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:${LD_LIBRARY_PATH:-}
export SHMEM_UID_SESSION_ID=127.0.0.1:${port}
if (( perf_mode != 0 )); then
    export SHMEM_CYCLE_PROF_PE="${prof_pe}"
else
    unset SHMEM_CYCLE_PROF_PE || true
fi

ipport="tcp://127.0.0.1:${port}"
pids=()

# A rank-local failure can leave peer processes blocked in a collective.
terminate_children() {
    local child_pid
    while read -r child_pid; do
        [[ -n "${child_pid}" ]] && kill "${child_pid}" 2>/dev/null || true
    done < <(jobs -pr)
    wait 2>/dev/null || true
}

trap terminate_children EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for pe_id in $(seq 0 $((pes - 1))); do
    "${binary}" "${pes}" "${pe_id}" "${ipport}" "${first_npu}" "${elements}" "${iterations}" "${aiv_count}" \
        "${dtype}" "${perf_mode}" "${perf_csv}" "${warmup_count}" "${mode}" "${tailcut_ratio}" &
    pids+=("$!")
done

active_pids="${pids[*]}"
while [[ -n "${active_pids}" ]]; do
    running_pids=" $(jobs -pr | tr '\n' ' ') "
    next_pids=""
    read -r -a active_pid_list <<< "${active_pids}"
    for pid in "${active_pid_list[@]}"; do
        if [[ "${running_pids}" == *" ${pid} "* ]]; then
            next_pids="${next_pids} ${pid}"
        elif wait "${pid}"; then
            :
        else
            exit_status=$?
            exit "${exit_status}"
        fi
    done
    active_pids="${next_pids# }"
    if [[ -n "${active_pids}" ]]; then
        sleep 0.05
    fi
done
trap - EXIT INT TERM

python3 "${script_dir}/check_result.py" \
    --pes "${pes}" \
    --elements "${elements}" \
    --tp-size "${tp_size}" \
    --dtype "${dtype}"
