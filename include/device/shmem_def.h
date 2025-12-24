/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_DEF_H
#define SHMEM_DEVICE_DEF_H

#include "kernel_operator.h"
#include "host_device/shmem_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Standard Atomic Add Types and Names
 *
 * |NAME       | TYPE      |
 * |-----------|-----------|
 * |half       | half      |
 * |float      | float     |
 * |int8       | int8      |
 * |int16      | int16     |
 * |int32      | int32     |
 */
#define ACLSHMEM_TYPE_FUNC_ATOMIC_INT(FUNC) \
    FUNC(int8, int8_t, ATOMIC_S8);          \
    FUNC(int16, int16_t, ATOMIC_S16);       \
    FUNC(int32, int32_t, ATOMIC_S32)

#define ACLSHMEM_TYPE_FUNC_ATOMIC_FLOAT(FUNC) \
    FUNC(half, half, ATOMIC_F16);             \
    FUNC(float, float, ATOMIC_F32)

#define ACLSHMEM_TEST_TYPE_FUNC(FUNC) \
    FUNC(float, float);               \
    FUNC(int8, int8_t);               \
    FUNC(int16, int16_t);             \
    FUNC(int32, int32_t);             \
    FUNC(int64, int64_t);             \
    FUNC(uint8, uint8_t);             \
    FUNC(uint16, uint16_t);           \
    FUNC(uint32, uint32_t);           \
    FUNC(uint64, uint64_t);           \
    FUNC(char, char)

constexpr uint64_t ACLSHMEM_INTERNAL_UB_BUF_START_ADDR = 188 * 1024;
constexpr uint32_t UB_ALIGN_SIZE = 32;
constexpr uint32_t ACLSHMEM_NUM_CQE_PER_POLL_CQ = 100;

/**
 * @addtogroup group_structs
 * @{
*/
/**
 * @struct non_contiguous_copy_param
 * @brief Non-Contiguous Datacopy Param.
 *
 * - uint32_t repeat: Data move times
 * - uint32_t length: Data move unit length
 * - uint32_t src_ld: Src data leading dimension. Interval between the head of the repeat and the
 *   head of the following repeat.
 * - uint32_t dst_ld: Dst data leading dimension.
*/
struct non_contiguous_copy_param {
    uint32_t repeat;
    uint32_t length;
    uint32_t src_ld;
    uint32_t dst_ld;
};

/**@} */ // end of group_structs

#ifdef __cplusplus
}
#endif

#endif