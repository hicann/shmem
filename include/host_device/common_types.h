/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_TYPES_H
#define SHMEM_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @private
*/
#define SHMEM_GLOBAL __global__ __aicore__

/// \def SHMEM_DEVICE
/// \brief A macro that identifies a function on the device side.
#define SHMEM_DEVICE __attribute__((always_inline)) __aicore__ __inline__

/**
 * @addtogroup group_enums
 * @{
*/

/**
* @brief The state of the SHMEM host OP type.
*/
enum shmemi_op_t {
    SHMEMI_OP_PUT = 0,
    SHMEMI_OP_P,
    SHMEMI_OP_PUT_SIGNAL,
    SHMEMI_OP_GET,
    SHMEMI_OP_G,
    // SHMEMI_OP_FENCE,
    // SHMEMI_OP_AMO,
    // SHMEMI_OP_QUIET,
    // SHMEMI_OP_SENTINEL = INT_MAX,
};

/**
 * @brief Team's index.
*/
enum shmem_team_index_t {
    SHMEM_TEAM_INVALID = -1,
    SHMEM_TEAM_WORLD = 0
};

/**
 * @brief Data op engine type.
*/
enum data_op_engine_type_t {
    SHMEM_DATA_OP_MTE = 0x01,
    SHMEM_DATA_OP_SDMA = 0x02,
    SHMEM_DATA_OP_ROCE = 0x04,
};

/**
 * @brief signal ops, used by signaler in p2p synchronization
 */
enum {
    SHMEM_SIGNAL_SET,
    SHMEM_SIGNAL_ADD
};

/**
 * @brief signal compare ops, used by signalee in p2p synchronization
 */
enum {
    SHMEM_CMP_EQ = 0,
    SHMEM_CMP_NE,
    SHMEM_CMP_GT,
    SHMEM_CMP_GE,
    SHMEM_CMP_LT,
    SHMEM_CMP_LE
};

/**
 * @brief Reserved for future use.
 */
typedef struct {
    int num_contexts;
} shmem_team_config_t;

/**@} */ // end of group_enums

/**
 * @defgroup group_typedef Typedef
 * @{

*/
/**
 * @brief A typedef of int
*/
typedef int shmem_team_t;

/**@} */ // end of group_typedef

/**
 * @addtogroup group_structs
 * @{
*/
/**
 * @struct shmem_handle_t
 * @brief Handle info used for non-blocking API synchronization.
 *
 * - shmem_team_t team_id: Team ID used for synchronization.
*/
struct shmem_handle_t {
    shmem_team_t team_id;
};

/**@} */ // end of group_structs

#define SHMEM_MAX_RANKS 16384
#define SHMEM_MAX_TEAMS 2048
#define SHMEM_MAX_LOCAL_SIZE (40UL * 1024 * 1024 * 1024)

/* arch related */
#define SCALAR_DATA_CACHELINE_SIZE 64
#define L2_CACHELINE_SIZE 512

#define SHMEM_PAGE_SIZE (1024UL * 1024UL * 2)

#define ALIGH_TO(size, page) (((size) + (page) - 1) / (page) * (page))

/* synchonization related */
#define SHMEMI_SYNCBIT_SIZE SCALAR_DATA_CACHELINE_SIZE

// npu level sync
#define SYNC_LOG_MAX_RANKS 5 // ceil(log_{8}^{16384}) = 5
#define SHMEM_BARRIER_TG_DISSEM_KVAL 8
#define SYNC_ARRAY_SIZE (SHMEMI_SYNCBIT_SIZE * SYNC_LOG_MAX_RANKS * SHMEM_BARRIER_TG_DISSEM_KVAL)
#define SYNC_COUNTER_SIZE SHMEMI_SYNCBIT_SIZE
#define SYNC_POOL_SIZE (SYNC_ARRAY_SIZE * SHMEM_MAX_TEAMS)
#define SYNC_COUNTERS_SIZE (SYNC_COUNTER_SIZE * SHMEM_MAX_TEAMS)

// core level sync
#define SHMEM_MAX_AIV_PER_NPU 48
#define SHMEM_LOG_MAX_AIV_PER_NPU 6     // ceil(log_{2}^{48}) = 6
#define SHMEM_CORE_SYNC_POOL_SIZE (SHMEM_MAX_AIV_PER_NPU * SHMEM_LOG_MAX_AIV_PER_NPU * SHMEMI_SYNCBIT_SIZE)
#define SHMEM_CORE_SYNC_COUNTER_SIZE SHMEMI_SYNCBIT_SIZE

// Total extra
#define SHMEM_EXTRA_SIZE_UNALIGHED SYNC_POOL_SIZE
#define SHMEM_EXTRA_SIZE ALIGH_TO(SHMEM_EXTRA_SIZE_UNALIGHED, SHMEM_PAGE_SIZE)

// global_state
constexpr uint64_t DEVMM_SVM_MEM_START = 0x100000000000ULL;
constexpr uint64_t SVM_END_ADDR = 0x100000000000ULL + 0x80000000000ULL - (1UL << 30UL); // svm end
constexpr uint64_t GLOBAL_STATE_SIZE = 4UL * 1024UL * 1024UL; // global_state fixed length

// synchronization
typedef int32_t shmemi_sync_bit[SHMEMI_SYNCBIT_SIZE / sizeof(int32_t)];

// Team
typedef struct {
    int mype;           // team view, [0, size]
    int start;          // global view, [0, npes]
    int stride;         // global view, [1, npes - 1]
    int size;           // team view
    int team_idx;
} shmemi_team_t;

// mte_config
typedef struct {
    int64_t shmem_ub;        // __ubuf__ Ptr, Shmem memcpy needed.
    uint32_t ub_size;        // UB's Size, in Bytes.
    uint32_t event_id;       // TEventID, for Shmem memcpy sync.
} shmemi_mte_config_t;

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

    uint8_t topo_list[SHMEM_MAX_RANKS];
    size_t heap_size;

    shmemi_team_t *team_pools[SHMEM_MAX_TEAMS];

    // Using shmemi_sync_bit instead of basic types to shmemi_store flag,
    // avoiding concurrent write due to cacheline sharing.
    // Refer to shmemi_barrier.h for more details.
    // These members are 'shmemi_sync_bit *' types actully, but are defined as 'uint64_t' due to compiler restriction.
    uint64_t sync_pool;
    uint64_t sync_counter;
    uint64_t core_sync_pool;
    uint64_t core_sync_counter;

    bool is_shmem_initialized;
    bool is_shmem_created;

    shmemi_mte_config_t mte_config;
    uint64_t qp_info;
} shmemi_device_host_state_t;

// host only state
typedef struct {
    // typedef void *aclrtStream; as in https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/API/appdevgapi/aclcppdevg_03_1355.html
    void *default_stream;
    int8_t default_event_id;
    uint32_t default_block_num;
} shmemi_host_state_t;


#ifdef __cplusplus
}
#endif

#endif /* SHMEM_TYPES_H */