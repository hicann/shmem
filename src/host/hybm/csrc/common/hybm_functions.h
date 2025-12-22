/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEM_FABRIC_HYBRID_HYBM_FUNCTIONS_H
#define MEM_FABRIC_HYBRID_HYBM_FUNCTIONS_H

#include "hybm_define.h"
#include "hybm_types.h"
#include "hybm_logger.h"

namespace shm {
namespace hybm {
class Func {
public:
    static uint64_t MakeObjectMagic(uint64_t srcAddress);

private:
    const static uint64_t gMagicBits = 0xFFFFFFFFFF; /* get lower 40bits */
};

inline uint64_t Func::MakeObjectMagic(uint64_t srcAddress)
{
    return (srcAddress & gMagicBits) + UN40;
}
}
}

#endif // MEM_FABRIC_HYBRID_HYBM_FUNCTIONS_H
