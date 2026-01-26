/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEM_FABRIC_HYBRID_HYBM_DEFINE_H
#define MEM_FABRIC_HYBRID_HYBM_DEFINE_H

#include <netinet/in.h>
#include <cstdint>
#include <cstddef>

namespace shm {

constexpr uint64_t DEVICE_LARGE_PAGE_SIZE = 2UL * 1024UL * 1024UL;  // 大页的size, 2M
constexpr uint64_t HYBM_DEVICE_VA_START = 0x100000000000UL;         // NPU上的地址空间起始: 16T
constexpr uint64_t HYBM_DEVICE_VA_SIZE = 0x80000000000UL;           // NPU上的地址空间范围: 8T
constexpr uint64_t HYBM_DEVICE_VA_END = HYBM_DEVICE_VA_START + HYBM_DEVICE_VA_SIZE;
constexpr uint64_t SVM_END_ADDR = HYBM_DEVICE_VA_START + HYBM_DEVICE_VA_SIZE - (1UL << 30UL); // svm的结尾虚拟地址
constexpr uint64_t HYBM_DEVICE_PRE_META_SIZE = 128UL; // 128B
constexpr uint64_t HYBM_DEVICE_GLOBAL_META_SIZE = HYBM_DEVICE_PRE_META_SIZE; // 128B
constexpr uint64_t HYBM_ENTITY_NUM_MAX = 511UL; // entity最大数量
constexpr uint64_t HYBM_DEVICE_META_SIZE = HYBM_DEVICE_PRE_META_SIZE * HYBM_ENTITY_NUM_MAX
    + HYBM_DEVICE_GLOBAL_META_SIZE; // 64K

constexpr uint64_t HYBM_DEVICE_USER_CONTEXT_PRE_SIZE = 64UL * 1024UL; // 64K
constexpr uint64_t HYBM_DEVICE_INFO_SIZE = HYBM_DEVICE_USER_CONTEXT_PRE_SIZE * HYBM_ENTITY_NUM_MAX
    + HYBM_DEVICE_META_SIZE; // 元数据+用户context,总大小32M, 对齐DEVICE_LARGE_PAGE_SIZE
constexpr uint64_t HYBM_DEVICE_META_ADDR = SVM_END_ADDR - HYBM_DEVICE_INFO_SIZE;
constexpr uint64_t HYBM_DEVICE_USER_CONTEXT_ADDR = HYBM_DEVICE_META_ADDR + HYBM_DEVICE_META_SIZE;
constexpr uint32_t ACL_MEMCPY_HOST_TO_HOST = 0;
constexpr uint32_t ACL_MEMCPY_HOST_TO_DEVICE = 1;
constexpr uint32_t ACL_MEMCPY_DEVICE_TO_HOST = 2;
constexpr uint32_t ACL_MEMCPY_DEVICE_TO_DEVICE = 3;

constexpr uint64_t DEVMM_HEAP_SIZE = (1UL << 30UL);     // same definition in devmm

constexpr uint64_t HYBM_HOST_CONN_START_ADDR = 0x30000000000UL;  // 48T
constexpr uint64_t HYBM_HOST_CONN_ADDR_SIZE = 0x100000000000UL;  // 16T

constexpr uint64_t HYBM_GVM_START_ADDR = 0x280000000000UL;      // 40T
constexpr uint64_t HYBM_GVM_END_ADDR = 0xA80000000000UL;        // 168T

inline bool IsVirtualAddressNpu(uint64_t address)
{
    return (address >= HYBM_DEVICE_VA_START && address < (HYBM_DEVICE_VA_START + HYBM_DEVICE_VA_SIZE));
}

inline bool IsVirtualAddressNpu(const void *address)
{
    return IsVirtualAddressNpu(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address)));
}

inline uint64_t Valid48BitsAddress(uint64_t address)
{
    return address & 0xffffffffffffUL;
}

inline const void *Valid48BitsAddress(const void *address)
{
    uint64_t addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address));
    return reinterpret_cast<const void *>(static_cast<uintptr_t>(Valid48BitsAddress(addr)));
}

inline void *Valid48BitsAddress(void *address)
{
    uint64_t addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address));
    return reinterpret_cast<void *>(static_cast<uintptr_t>(Valid48BitsAddress(addr)));
}

struct HybmDeviceMeta {
    uint32_t entityId;
    uint32_t rankId;
    uint32_t rankSize;
    uint32_t extraContextSize;
    uint64_t symmetricSize;
    uint64_t qpInfoAddress;
    uint64_t reserved[12]; // total 128B, equal HYBM_DEVICE_PRE_META_SIZE
};

typedef struct drv_mem_handle {
    int id;
    uint32_t side;
    uint32_t sdid;
    uint32_t devid;
    uint32_t module_id;
    uint64_t pg_num;

    uint32_t pg_type;
    uint32_t phy_mem_type;
} drv_mem_handle_t;

typedef enum {
    MEM_HANDLE_TYPE_NONE = 0x0,
    MEM_HANDLE_TYPE_FABRIC = 0x1,
    MEM_HANDLE_TYPE_MAX = 0x2,
} drv_mem_handle_type;

#define MEM_SHARE_HANDLE_LEN 128
struct MemShareHandle {
    uint8_t share_info[MEM_SHARE_HANDLE_LEN];
};

enum drv_mem_side {
    MEM_HOST_SIDE = 0,
    MEM_DEV_SIDE,
    MEM_HOST_NUMA_SIDE,
    MEM_MAX_SIDE
};

enum drv_mem_pg_type {
    MEM_NORMAL_PAGE_TYPE = 0,
    MEM_HUGE_PAGE_TYPE,
    MEM_GIANT_PAGE_TYPE,
    MEM_MAX_PAGE_TYPE
};

enum drv_mem_type {
    MEM_HBM_TYPE = 0,
    MEM_DDR_TYPE,
    MEM_P2P_HBM_TYPE,
    MEM_P2P_DDR_TYPE,
    MEM_TS_DDR_TYPE,
    MEM_MAX_TYPE
};

struct drv_mem_prop {
    uint32_t side;       /* enum drv_mem_side */
    uint32_t devid;
    uint32_t module_id;  /* module id defines in ascend_hal_define.h */

    uint32_t pg_type;    /* enum drv_mem_pg_type */
    uint32_t mem_type;   /* enum drv_mem_type */
    uint64_t reserve;
};

typedef enum {
    MEM_ALLOC_GRANULARITY_MINIMUM = 0x0,
    MEM_ALLOC_GRANULARITY_RECOMMENDED,
    MEM_ALLOC_GRANULARITY_INVALID,
} drv_mem_granularity_options;

enum ShareHandleAttrType {
    SHR_HANDLE_ATTR_NO_WLIST_IN_SERVER = 0,
    SHR_HANDLE_ATTR_TYPE_MAX
};

#define SHR_HANDLE_WLIST_ENABLE     0x0
#define SHR_HANDLE_NO_WLIST_ENABLE  0x1
struct ShareHandleAttr {
    unsigned int enableFlag;     /* wlist enable: 0 no wlist enable: 1 */
    unsigned int rsv[8];
};

#define MEM_RSV_TYPE_DEVICE_SHARE_BIT   8 /* mmap va in all opened devices, exclude host to avoid va conflict */
#define MEM_RSV_TYPE_DEVICE_SHARE       (0x1u << MEM_RSV_TYPE_DEVICE_SHARE_BIT)
#define MEM_RSV_TYPE_REMOTE_MAP_BIT     9 /* this va only map remote addr, not create double page table */
#define MEM_RSV_TYPE_REMOTE_MAP         (0x1u << MEM_RSV_TYPE_REMOTE_MAP_BIT)

}  // namespace shm

#endif  // MEM_FABRIC_HYBRID_HYBM_DEFINE_H
