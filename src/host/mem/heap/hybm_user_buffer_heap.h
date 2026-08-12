/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEM_FABRIC_HYBRID_HYBM_USER_BUFFER_HEAP_H
#define MEM_FABRIC_HYBRID_HYBM_USER_BUFFER_HEAP_H

#include <cstdint>
#include <vector>

#include "acl/acl_rt.h"

namespace shm {

enum class UserBufferHandleOwnership : uint8_t {
    CALLER,
    RETAINED,
};

struct UserBufferEntry {
    void* source_base{nullptr};
    aclrtDrvMemHandle mem_handle{nullptr};
    uint64_t size{0};
    uint64_t segment_offset{0};
    aclrtPhysicalMemProp property{};
#ifdef HAS_ACLRT_MEM_FABRIC_HANDLE
    bool has_fabric_handle{false};
    aclrtMemFabricHandle fabric_handle{};
#endif
    UserBufferHandleOwnership handle_ownership{UserBufferHandleOwnership::CALLER};
};

struct UserBufferHeapInput {
    UserBufferHeapInput() = default;
    ~UserBufferHeapInput();
    UserBufferHeapInput(const UserBufferHeapInput&) = delete;
    UserBufferHeapInput& operator=(const UserBufferHeapInput&) = delete;

    std::vector<UserBufferEntry> entries;
};

} // namespace shm

#endif // MEM_FABRIC_HYBRID_HYBM_USER_BUFFER_HEAP_H
