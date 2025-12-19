/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "dl_api.h"
#include "dl_acl_api.h"
#include "dl_hal_api.h"
#include "dl_hccp_api.h"

namespace ock {
namespace mf {

Result DlApi::LoadLibrary(const std::string &libDirPath)
{
    auto result = DlAclApi::LoadLibrary(libDirPath);
    if (result != BM_OK) {
        return result;
    }

    result = DlHalApi::LoadLibrary();
    if (result != BM_OK) {
        DlAclApi::CleanupLibrary();
        return result;
    }

    return BM_OK;
}

void DlApi::CleanupLibrary()
{
    DlHccpApi::CleanupLibrary();
    DlAclApi::CleanupLibrary();
    DlHalApi::CleanupLibrary();
}

Result DlApi::LoadExtendLibrary(DlApiExtendLibraryType libraryType)
{
    if (libraryType == DL_EXT_LIB_DEVICE_RDMA) {
        return DlHccpApi::LoadLibrary();
    }

    return BM_OK;
}

}
}