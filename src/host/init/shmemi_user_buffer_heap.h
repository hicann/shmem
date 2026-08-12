/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHMEMI_USER_BUFFER_HEAP_H
#define SHMEMI_USER_BUFFER_HEAP_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "host/shmem_host_def.h"
#include "hybm_user_buffer_heap.h"
#include "utils/shmemi_host_types.h"

namespace shm {

inline constexpr size_t kMaxBuffers = 1024;
inline constexpr size_t kMaxBufferMetadataBytes = 64ULL * 1024ULL * 1024ULL;

struct UserBufferHeapLayoutHeader {
    int32_t local_status;
    uint32_t buffer_count;
    uint64_t external_bytes;
    uint64_t allocatable_bytes;
    uint64_t heap_size;
};

static_assert(std::is_trivially_copyable_v<UserBufferHeapLayoutHeader>);

} // namespace shm

int32_t aclshmemi_collective_status_gate(int32_t local_status, int32_t npes, aclshmemi_bootstrap_handle_t* boot_handle);

#endif // SHMEMI_USER_BUFFER_HEAP_H
