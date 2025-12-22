/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEM_FABRIC_HYBRID_HYBM_TYPES_H
#define MEM_FABRIC_HYBRID_HYBM_TYPES_H

#include <cstdint>
#include <memory>

namespace shm {
namespace hybm {
using Result = int32_t;

enum BErrorCode : int32_t {
    BM_OK = 0,
    BM_ERROR = -1,
    BM_INVALID_PARAM = -2,
    BM_MALLOC_FAILED = -3,
    BM_NEW_OBJECT_FAILED = -4,
    BM_FILE_NOT_ACCESS = -5,
    BM_DL_FUNCTION_FAILED = -6,
    BM_TIMEOUT = -7,
    BM_UNDER_API_UNLOAD = -8,
    BM_NOT_INITIALIZED = -9,
    BM_NOT_SUPPORTED = -100,
};

constexpr uint32_t UN40 = 40;
} // namespace hybm
} // namespace shm

#endif // MEM_FABRIC_HYBRID_HYBM_TYPES_H
