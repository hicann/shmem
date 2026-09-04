#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# HBM 内存泄漏检查脚本, 由 run.sh 在 UT 运行前后各调用一次:
#   UT 运行前: hbm_leak_check.sh before <first_npu> <gnpu_num> <state_file>  采样基线并保存到 state_file
#   UT 运行后: hbm_leak_check.sh after  <first_npu> <gnpu_num> <state_file>  再次采样并与基线对比
# 退出码:
#   before: 0=基线采样完整  1=部分卡采样失败(仅警告, 不阻塞 UT)
#   after:  0=未检测到泄漏  1=检测到内存泄漏  2=参数错误
# 环境变量:
#   HBM_CHECK_WAIT_SECONDS: UT 结束后等待显存释放的秒数, 默认 30
#   ASCEND_RT_VISIBLE_DEVICES: 设置了可见卡(如 "2" 或 "4,5")时, <first_npu> 起始的
#     逻辑卡号会被映射为可见列表中的物理卡号后再查询 npu-smi, 只检查 UT 实际使用的卡

readonly THRESHOLD_MB=10
readonly CLEANUP_WAIT_SECONDS="${HBM_CHECK_WAIT_SECONDS:-30}"

# 解析 npu-smi info, 按芯片顺序输出 "芯片序号 已用HBM(MB)", 解析失败时值为 -1。
# 兼容 910B / 910C / 950 / 950DT 等机型的输出格式:
#   - 不依赖 Bus-Id 定位(Bus-Id 可能为空或为 NA), 按表格结构解析:
#     每个芯片占两行, 第二行最后一个字段的最后一组 "used / total" 即 HBM-Usage(MB)
#   - 行首可能带空格(如 950DT), 解析前先跳过行首空白
#   - 进入进程信息表(Process id / Process name)后不再有 HBM 数据, 停止解析
sample_all_cards()
{
    npu-smi info 2>/dev/null | awk '
        /Process id|Process name/ { exit }
        {
            line = $0
            sub(/^[ \t\r]+/, "", line)
            if (line == "") { next }
            first = substr(line, 1, 1)
            if (first == "+") { row = 0; next }
            if (line ~ /HBM-Usage|Health/) { row = 0; next }
            if (first != "|") { next }
            row++
            if (row < 2) { next }
            used = -1
            nf = split(line, field, "|")
            if (nf >= 3 && match(field[nf - 1], /[0-9]+[ \t]*\/[ \t]*[0-9]+[ \t]*$/)) {
                pair = substr(field[nf - 1], RSTART, RLENGTH)
                split(pair, value, /[ \t]*\/[ \t]*/)
                used = value[1] + 0
            }
            printf "%d %d\n", chip, used
            chip++
            row = 0
        }
    '
}

# 从采样结果中取指定卡的 HBM 用量, 取不到时输出 -1
get_card_hbm()
{
    local sample=$1 card=$2 used
    used=$(printf '%s\n' "$sample" | awk -v card="$card" '$1 == card { print $2; exit }')
    if [ -z "$used" ]; then
        echo "-1"
    else
        echo "$used"
    fi
}

# 将 UT 使用的逻辑卡号映射为 npu-smi 中的物理卡号, 映射失败时输出空串:
#   - 未设置 ASCEND_RT_VISIBLE_DEVICES 时, 逻辑卡号即物理卡号
#   - 设置了 ASCEND_RT_VISIBLE_DEVICES(如 "2" 或 "4,5")时,
#     逻辑卡号 n 对应可见列表中的第 n+1 个物理卡号
map_card()
{
    local logical=$1
    if [ -z "$ASCEND_RT_VISIBLE_DEVICES" ]; then
        echo "$logical"
        return
    fi
    echo "$ASCEND_RT_VISIBLE_DEVICES" | awk -v n="$logical" '
        { gsub(/[ \t]/, ""); split($0, d, ","); if ((n + 1) in d) print d[n + 1] }
    '
}

cmd_before()
{
    local first_npu=$1 gnpu_num=$2 state_file=$3
    echo "=========================================="
    echo "HBM Memory Check - BEFORE UT"
    echo "=========================================="

    local sample card physical used fail=0
    sample=$(sample_all_cards)
    if [ -n "$sample" ]; then
        printf '%s\n' "$sample" > "$state_file"
    fi

    for ((card = first_npu; card < first_npu + gnpu_num; card++)); do
        physical=$(map_card "$card")
        if [ -z "$physical" ]; then
            fail=1
            echo "WARNING: Card $card is not in ASCEND_RT_VISIBLE_DEVICES"
            echo "WARNING: ('${ASCEND_RT_VISIBLE_DEVICES}'), skip check for it"
            continue
        fi
        used=$(get_card_hbm "$sample" "$physical")
        if [ "$used" -ge 0 ]; then
            echo "Card $physical: HBM Used = $used MB"
        else
            fail=1
            echo "WARNING: Failed to get HBM usage for card $physical"
        fi
    done

    if [ "$fail" -ne 0 ]; then
        echo "WARNING: Failed to sample HBM baseline for one or more cards. Leak check may be incomplete."
    fi
    echo "=========================================="
    return "$fail"
}

cmd_after()
{
    local first_npu=$1 gnpu_num=$2 state_file=$3
    echo "Waiting ${CLEANUP_WAIT_SECONDS} seconds for cleanup..."
    sleep "$CLEANUP_WAIT_SECONDS"
    echo "=========================================="
    echo "HBM Memory Check - AFTER UT"
    echo "=========================================="

    local sample
    sample=$(sample_all_cards)

    local card physical used before diff has_leak=0 has_warning=0
    for ((card = first_npu; card < first_npu + gnpu_num; card++)); do
        physical=$(map_card "$card")
        if [ -z "$physical" ]; then
            has_warning=1
            echo "WARNING: Card $card is not in ASCEND_RT_VISIBLE_DEVICES"
            echo "WARNING: ('${ASCEND_RT_VISIBLE_DEVICES}'), skip check for it"
            continue
        fi
        used=$(get_card_hbm "$sample" "$physical")
        if [ "$used" -lt 0 ]; then
            has_warning=1
            echo "WARNING: Failed to get HBM usage for card $physical"
            continue
        fi

        before=$(awk -v card="$physical" '$1 == card { print $2; exit }' "$state_file" 2>/dev/null)
        if [ -z "$before" ] || [ "$before" -lt 0 ]; then
            has_warning=1
            echo "Card $physical: HBM Used = $used MB (no baseline - check skipped)"
            continue
        fi

        diff=$((used - before))
        echo "Card $physical:"
        echo "  Before: $before MB"
        echo "  After:  $used MB"
        echo "  Diff:   $diff MB"
        if [ "$diff" -gt "$THRESHOLD_MB" ]; then
            echo "  Status: ERROR: Memory leak detected! (exceeds $THRESHOLD_MB MB threshold)"
            has_leak=1
        elif [ "$diff" -lt "-$THRESHOLD_MB" ]; then
            echo "  Status: Memory freed ($((-diff)) MB released)"
        else
            echo "  Status: No leak detected (within $THRESHOLD_MB MB tolerance)"
        fi
        echo ""
    done

    rm -f "$state_file"

    echo "=========================================="
    if [ "$has_warning" -ne 0 ]; then
        echo "WARNING: HBM check incomplete - baseline unavailable or sampling failed for one or more cards."
        echo ""
        echo "Possible causes:"
        echo "  1. npu-smi tool is not available or failed to run."
        echo "     Please verify: npu-smi info"
        echo "  2. HBM baseline sampling before UT failed."
        echo ""
    fi
    if [ "$has_leak" -ne 0 ]; then
        echo "ERROR: Memory leak detected on one or more cards!"
        echo ""
        echo "Possible causes:"
        echo "  1. Other programs are using the NPU cards concurrently."
        echo "     Please check if other processes are using the cards."
        echo "  2. Memory leak in the code under test."
        echo "     Please review the code for potential memory leaks."
    else
        echo "Overall: No memory leak detected"
    fi
    echo "=========================================="
    return "$has_leak"
}

main()
{
    if [ $# -ne 4 ]; then
        echo "Usage: $0 {before|after} <first_npu> <gnpu_num> <state_file>" >&2
        exit 2
    fi
    case "$1" in
        before)
            cmd_before "$2" "$3" "$4"
            ;;
        after)
            cmd_after "$2" "$3" "$4"
            ;;
        *)
            echo "Usage: $0 {before|after} <first_npu> <gnpu_num> <state_file>" >&2
            exit 2
            ;;
    esac
}

main "$@"
