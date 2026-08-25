#!/bin/bash

set -ex

function check_docs_changes() {
    # ========== 新增：检测变更范围，仅文档（docs/ 或 .md）时跳过 UT ==========
    # 获取变更文件列表（与 target_branch 比较）
    CHANGED_FILES=""
    if [ -n "${target_branch}" ]; then
        # CI 场景：与目标分支比较
        CHANGED_FILES=$(git diff --name-only origin/${target_branch}...HEAD 2>/dev/null || true)
    else
        # 后备方案：与当前分支的上游（通常是 origin/master）比较
        CHANGED_FILES=$(git diff --name-only origin/master...HEAD 2>/dev/null || true)
        if [ -z "${CHANGED_FILES}" ]; then
            CHANGED_FILES=$(git diff --name-only HEAD~1 2>/dev/null || true)
        fi
    fi

    # 检查变更文件是否全部为文档（docs/ 开头 或 .md 结尾）
    if [ -n "${CHANGED_FILES}" ]; then
        ONLY_DOCS=true
        for file in ${CHANGED_FILES}; do
            # 判断：不以 docs/ 开头 且 不以 .md 结尾 → 非文档变更
            if [[ ! "${file}" =~ ^docs/ ]] && [[ ! "${file}" =~ \.md$ ]]; then
                ONLY_DOCS=false
                break
            fi
        done
        if [ "${ONLY_DOCS}" = true ]; then
            echo "----------------------------------------"
            echo "Only documentation files (docs/ or .md) changed. Skipping all UT."
            echo "----------------------------------------"
            exit 0   # 正常退出，不执行后续构建和测试
        fi
    fi
}

function env_info() {
    echo "=======================================================/usr/local/Ascend/==============================="
    ls -la /usr/local/Ascend/

    echo "=======================================================/dev/davinci====================================="
    ls -la /dev/davinci*

    echo "=======================================================driver info====================================="
    cat /usr/local/Ascend/driver/version.info

    npu-smi info
}
