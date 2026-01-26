/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef DL_VER_UTIL_INCLUDE_H
#define DL_VER_UTIL_INCLUDE_H

#include <string>

#include "host/shmem_host_def.h"

enum HybmGvaVersion : uint32_t {
    HYBM_GVA_V1 = 0,
    HYBM_GVA_V2 = 1,
    HYBM_GVA_V3 = 2,
    HYBM_GVA_UNKNOWN
};

enum DeviceSystemInfoType {
    INFO_TYPE_PHY_CHIP_ID = 18,
    INFO_TYPE_PHY_DIE_ID,
    INFO_TYPE_SDID = 26,
    INFO_TYPE_SERVER_ID,
    INFO_TYPE_SCALE_TYPE,
    INFO_TYPE_SUPER_POD_ID,
    INFO_TYPE_ADDR_MODE,
};

HybmGvaVersion HybmGetGvaVersion();

static bool DriverVersionCheck(const std::string &ver);

int32_t HalGvaPrecheck(void);

#endif  // DL_VER_UTIL_INCLUDE_H