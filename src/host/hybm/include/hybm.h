/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEM_FABRIC_HYBRID_HYBM_H
#define MEM_FABRIC_HYBRID_HYBM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize hybrid big memory library
 *
 * @param deviceId         [in] npu device id
 * @param flags            [in] optional flags
 * @return 0 if successful
 */
int32_t hybm_init(uint16_t deviceId, uint64_t flags);

/**
 * @brief UnInitialize hybrid big memory library
 */
void hybm_uninit(void);

#ifdef __cplusplus
}
#endif

#endif // MEM_FABRIC_HYBRID_HYBM_H
