# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

add_subdirectory(host)
add_subdirectory(device)

add_executable(aclshmem_fuzztest $<TARGET_OBJECTS:aclshmem_fuzztest_host>)
target_link_directories(aclshmem_fuzztest PRIVATE
    ${PROJECT_SOURCE_DIR}/3rdparty/googletest/lib
    ${PROJECT_SOURCE_DIR}/3rdparty/secodefuzz/lib
)
target_link_libraries(aclshmem_fuzztest PRIVATE aclshmem_fuzztest_device gtest gcov Secodefuzz)
