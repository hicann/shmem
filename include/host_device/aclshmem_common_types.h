/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEM_TYPES_H
#define ACLSHMEM_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @private
*/
#define ACLSHMEM_GLOBAL __global__ __aicore__

/// \def ACLSHMEM_DEVICE
/// \brief A macro that identifies a function on the device side.
#define ACLSHMEM_DEVICE __attribute__((always_inline)) __aicore__ __inline__

/**
 * @addtogroup group_enums
 * @{
*/

/**
* @brief The state of the ACLSHMEM host OP type.
*/
enum aclshmemi_op_t {
    ACLSHMEMI_OP_PUT = 0,
    ACLSHMEMI_OP_P,
    ACLSHMEMI_OP_PUT_SIGNAL,
    ACLSHMEMI_OP_GET,
    ACLSHMEMI_OP_G,
    // ACLSHMEMI_OP_FENCE,
    // ACLSHMEMI_OP_AMO,
    // ACLSHMEMI_OP_QUIET,
    // ACLSHMEMI_OP_SENTINEL = INT_MAX,
};

/**
 * @brief Team's index.
*/
enum aclshmem_team_index_t {
    ACLSHMEM_TEAM_INVALID = -1,
    ACLSHMEM_TEAM_WORLD = 0
};

/**
 * @brief Data op engine type.
*/
enum data_op_engine_type_t {
    ACLSHMEM_DATA_OP_MTE = 0x01,
    ACLSHMEM_DATA_OP_SDMA = 0x02,
    ACLSHMEM_DATA_OP_ROCE = 0x04,
};

/**
 * @brief Memory type of NPU or host
 */
enum aclshmem_mem_type_t {
    HOST_SIDE = 0, // shmem申请地址为host侧
    DEVICE_SIDE    // shmem申请地址为device侧
};

/**
 * @brief signal ops, used by signaler in p2p synchronization
 */
enum {
    ACLSHMEM_SIGNAL_SET,
    ACLSHMEM_SIGNAL_ADD
};

/**
 * @brief signal compare ops, used by signalee in p2p synchronization
 */
enum {
    ACLSHMEM_CMP_EQ = 0,
    ACLSHMEM_CMP_NE,
    ACLSHMEM_CMP_GT,
    ACLSHMEM_CMP_GE,
    ACLSHMEM_CMP_LT,
    ACLSHMEM_CMP_LE
};

/**@} */ // end of group_enums

/**
 * @defgroup group_typedef Typedef
 * @{

*/
/**
 * @brief A typedef of int
*/
typedef int aclshmem_team_t;

typedef uint64_t aclshmemx_team_uniqueid_t;

/**@} */ // end of group_typedef

/**
 * @addtogroup group_structs
 * @{
*/
/**
 * @struct aclshmem_handle_t
 * @brief Handle info used for non-blocking API synchronization.
 *
 * - aclshmem_team_t team_id: Team ID used for synchronization.
*/
struct aclshmem_handle_t {
    aclshmem_team_t team_id;
};

/**@} */ // end of group_structs

#define ACLSHMEM_MAX_PES 16384
#define ACLSHMEM_MAX_TEAMS 2048
#define ACLSHMEM_MAX_LOCAL_SIZE (40UL * 1024 * 1024 * 1024)

#define TEAM_CONFIG_PADDING 48

/* arch related */
#define SCALAR_DATA_CACHELINE_SIZE 64
#define L2_CACHELINE_SIZE 512

#define ACLSHMEM_PAGE_SIZE (1024UL * 1024UL * 2)

#define ALIGH_TO(size, page) (((size) + (page) - 1) / (page) * (page))

/* synchonization related */
#define ACLSHMEMI_SYNCBIT_SIZE SCALAR_DATA_CACHELINE_SIZE

// npu level sync
#define SYNC_LOG_MAX_PES 5 // ceil(log_{8}^{16384}) = 5
#define ACLSHMEM_BARRIER_TG_DISSEM_KVAL 8
#define SYNC_ARRAY_SIZE (ACLSHMEMI_SYNCBIT_SIZE * SYNC_LOG_MAX_PES * ACLSHMEM_BARRIER_TG_DISSEM_KVAL)
#define SYNC_COUNTER_SIZE ACLSHMEMI_SYNCBIT_SIZE
#define SYNC_POOL_SIZE (SYNC_ARRAY_SIZE * ACLSHMEM_MAX_TEAMS)
#define SYNC_COUNTERS_SIZE (SYNC_COUNTER_SIZE * ACLSHMEM_MAX_TEAMS)

// core level sync
#define ACLSHMEM_MAX_AIV_PER_NPU 48
#define ACLSHMEM_LOG_MAX_AIV_PER_NPU 6     // ceil(log_{2}^{48}) = 6
#define ACLSHMEM_CORE_SYNC_POOL_SIZE (ACLSHMEM_MAX_AIV_PER_NPU * ACLSHMEM_LOG_MAX_AIV_PER_NPU * ACLSHMEMI_SYNCBIT_SIZE)
#define ACLSHMEM_CORE_SYNC_COUNTER_SIZE ACLSHMEMI_SYNCBIT_SIZE

// Total extra
#define ACLSHMEM_EXTRA_SIZE_UNALIGHED SYNC_POOL_SIZE
#define ACLSHMEM_EXTRA_SIZE ALIGH_TO(ACLSHMEM_EXTRA_SIZE_UNALIGHED, ACLSHMEM_PAGE_SIZE)

// global_state
constexpr uint64_t SVM_START_ADDR = 0x100000000000ULL;
constexpr uint64_t SVM_END_ADDR = 0x100000000000ULL + 0x80000000000ULL - (1UL << 30UL); // svm end
constexpr uint64_t GLOBAL_STATE_SIZE = 4UL * 1024UL * 1024UL; // global_state fixed length

// synchronization
typedef int32_t aclshmemi_sync_bit[ACLSHMEMI_SYNCBIT_SIZE / sizeof(int32_t)];

/**
 * @brief 分组管理配置
 */
typedef struct {
    int32_t version;                      /// 版本号
    int32_t num_contexts;                 /// 预留字段
    aclshmemx_team_uniqueid_t uniqueid;   /// 在分组中的唯一id
    char padding[TEAM_CONFIG_PADDING];    /// 用于补齐结构体到64字节
} aclshmem_team_config_t;
static_assert(sizeof(aclshmem_team_config_t) == 64, "aclshmem_team_config_t must be 64 bytes.");

// Team
typedef struct {
    int mype;           // team view, [0, size]
    int start;          // global view, [0, npes]
    int stride;         // global view, [1, npes - 1]
    int size;           // team view
    int team_idx;
    aclshmem_team_config_t config;
    int32_t pe_mapping[2 * ACLSHMEM_MAX_PES];
} aclshmemx_team_t;

// mte_config
typedef struct {
    int64_t aclshmem_ub;        // __ubuf__ Ptr, Shmem memcpy needed.
    uint32_t ub_size;        // UB's Size, in Bytes.
    uint32_t event_id;       // TEventID, for Shmem memcpy sync.
} aclshmem_mte_config_t;

// state
typedef struct {
    int version;
    int mype;
    int npes;
    void *heap_base;

    // Store All Devices' heap_base in Host.
    void **host_p2p_heap_base;
    void **host_rdma_heap_base;
    void **host_sdma_heap_base;

    // Store All Devices' heap_base in Device.
    void **device_p2p_heap_base;
    void **device_rdma_heap_base;
    void **device_sdma_heap_base;

    uint8_t topo_list[ACLSHMEM_MAX_PES];
    size_t heap_size;

    aclshmemx_team_t *team_pools[ACLSHMEM_MAX_TEAMS];

    // Using aclshmemi_sync_bit instead of basic types to aclshmemi_store flag,
    // avoiding concurrent write due to cacheline sharing.
    // Refer to aclshmemi_barrier.h for more details.
    // These members are 'shmemi_sync_bit *' types actully, but are defined as 'uint64_t' due to compiler restriction.
    uint64_t sync_pool;
    uint64_t sync_counter;
    uint64_t core_sync_pool;
    uint64_t core_sync_counter;

    bool is_aclshmem_initialized;
    bool is_aclshmem_created;

    aclshmem_mte_config_t mte_config;
    uint64_t qp_info;
} aclshmem_device_host_state_t;

// host only state
typedef struct {
    // typedef void *aclrtStream; as in https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/API/appdevgapi/aclcppdevg_03_1355.html
    void *default_stream;
    int8_t default_event_id;
    uint32_t default_block_num;
} aclshmem_host_state_t;

#ifdef __cplusplus
}
#endif

#endif /* ACLSHMEM_TYPES_H */