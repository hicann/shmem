#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set_env_path="${BASH_SOURCE[0]}"
if [[ -f "$set_env_path" ]] && [[ "$(basename "$set_env_path")" == "set_env.sh" ]]; then
    aclshmem_path=$(cd $(dirname $set_env_path); pwd)
    export ACLSHMEM_HOME_PATH="$aclshmem_path"
    export LD_LIBRARY_PATH=$ACLSHMEM_HOME_PATH/shmem/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH
    export PATH=$ACLSHMEM_HOME_PATH/bin:$PATH
fi
# 是否有python扩展
if [[ -d "$ACLSHMEM_HOME_PATH/../examples/shared_lib/output" ]] && [[ -d "$ACLSHMEM_HOME_PATH/../examples/python_extension/output" ]]; then
    echo "Export the environment variable for the aclshmem python extension. "
    export LD_LIBRARY_PATH=$ACLSHMEM_HOME_PATH/../examples/shared_lib/output/lib:$ACLSHMEM_HOME_PATH/../examples/python_extension/output/lib:$LD_LIBRARY_PATH
fi