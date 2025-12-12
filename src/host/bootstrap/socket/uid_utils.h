/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHMEM_UID_UTILS_H
#define SHMEM_UID_UTILS_H


#include "utils/aclshmemi_logger.h"
#include <stdlib.h>   // malloc, free
#include <string.h>   // memset

template <typename T>
inline int bootstrap_calloc(T** ptr, size_t nelem, const char* file, int line) {
    if (ptr == nullptr || nelem == 0) {  // 校验输入：指针为空或元素数为 0 均为无效
        SHM_LOG_ERROR("Invalid arguments: ptr=" << ptr << ", nelem=" << nelem 
                             << " (" << file << ":" << line << ")");
        return SHMEM_BOOTSTRAP_ERROR;
    }
    size_t total_size = nelem * sizeof(T);  // 计算总内存大小
    void* p = malloc(total_size);
    if (p == nullptr) {
        SHM_LOG_ERROR("Allocation failed: " << total_size << " bytes (nelem=" << nelem 
                             << ") at " << file << ":" << line);
        return SHMEM_BOOTSTRAP_ERROR;
    }

    memset(p, 0, total_size);  // 内存清零
    *ptr = static_cast<T*>(p);  // 类型转换，赋值给输出指针

    // 调试日志：输出分配信息
    SHM_LOG_DEBUG("Allocated " << total_size << " bytes (" << nelem 
                          << " elements of " << sizeof(T) << " bytes) at " 
                          << static_cast<void*>(p) << " (" << file << ":" << line << ")");
    return SHMEM_SUCCESS;
}
#define SHMEM_BOOTSTRAP_CALLOC(ptr, nelem) \
    bootstrap_calloc((ptr), (nelem), __FILE__, __LINE__)


#define SHMEM_BOOTSTRAP_PTR_FREE(ptr)   \
    do {                                \
        if ((ptr) != NULL) {            \
            free(ptr);                  \
        }                               \
    } while (0)

#define SHMEM_CHECK_RET_CLOSE_SOCK(x, LOG_STR, SOCK)                                                                \
    do {                                                                                                            \
        int32_t check_ret = (x);                                                                                    \
        if (check_ret != 0) {                                                                                       \
            SHM_LOG_ERROR(" " << LOG_STR << " close sock " << #SOCK << " and return shmem error: " << check_ret);   \
            if ((&(SOCK)) != nullptr) {                                                                             \
                socket_close(&(SOCK));                                                                              \
            }                                                                                                       \
            return check_ret;                                                                                       \
        }                                                                                                           \
    } while (0)

#endif  //SHMEM_UID_UTILS_H