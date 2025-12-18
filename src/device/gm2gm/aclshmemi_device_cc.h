/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
This file provides device-side collective synchronization implementations, ensuring that:
1. ALL VEC CORES of all ranks of a team reach a sychonization point before doing subsequent operations.
2. All operations of ALL VEC CORES of all ranks of the team before the synchronization point are visible
   to ALL VEC CORES of all ranks of the team after the synchronization point.

*/

#ifndef ACLSHMEMI_DEVICE_CC_H
#define ACLSHMEMI_DEVICE_CC_H

#include "stdint.h"
#include "aclshmemi_device_mo.h"
#include "aclshmemi_device_p2p_sync.h"

#include "kernel_operator.h"

const int SHIFT_MULTIPLIER = 2;
constexpr uint64_t ACLSHMEM_DATA_CACHE_LINE_SIZE = 64;

ACLSHMEM_DEVICE
__gm__ aclshmemi_sync_bit *aclshmemi_get_core_sync_array();

ACLSHMEM_DEVICE
__gm__ aclshmemi_sync_bit *aclshmemi_get_core_sync_counter();

ACLSHMEM_DEVICE void aclshmemi_barrier_core_soft();

/* Level 1: barrier between vec cores (within a device) */
template<bool is_aiv_only = true>
ACLSHMEM_DEVICE void aclshmemi_barrier_core();

ACLSHMEM_DEVICE
__gm__ aclshmemi_sync_bit *aclshmemi_get_team_sync_array(aclshmem_team_t team_idx);

ACLSHMEM_DEVICE
__gm__ aclshmemi_sync_bit *aclshmemi_get_team_sync_counter(aclshmem_team_t team_idx);

/* Level 2: barrier between devices (within a host)

Dissemination Barrier

1. Algorithm process

The algorithm process could be separated into multiple rounds.
In each round, every participating rank signals its succeeding rank and waits its preceding rank's signal.
The distance of a rank and its successor increases exponentially with the round.

An 8-rank example is shown below:

           round 1            round 2            round 3
  rank 0  --------→  rank 1  --------→  rank 3  --------→  rank 7
  rank 1  --------→  rank 2  --------→  rank 4  --------→  rank 0
  rank 2  --------→  rank 3  --------→  rank 5  --------→  rank 1
  rank 3  --------→  rank 4  --------→  rank 6  --------→  rank 2
  rank 4  --------→  rank 5  --------→  rank 7  --------→  rank 3
  rank 5  --------→  rank 6  --------→  rank 0  --------→  rank 4
  rank 6  --------→  rank 7  --------→  rank 1  --------→  rank 5
  rank 7  --------→  rank 0  --------→  rank 2  --------→  rank 6

Refer to https://www.inf.ed.ac.uk/teaching/courses/ppls/BarrierPaper.pdf for more details.

2. Implementation details

Current implementation maintains an array of MAX_RANK_SIZE for each rank, with element of pos i indicating whether the
rank has received signal of rank i.
In each round, every rank writes remote array and check local array to decide whether this round has finished.
Once all rounds finished, barrier ends.

Theoretically, each element is writen by only 1 rank and read by self, involving only p2p synchronization.
However, separate elements may exist on the same cacheline, so that concurrent write
acctually happens and may cause wrong result.

For example:
a. rank n is waiting for rank n-1's signal (in round 1).
             ↑   n
--------------------------------------------
      ...  | 0 | 0 | ...
--------------------------------------------

b. rank n-1 reads rank n's array, and write the element at position n-1 (in round 1).
             ↓   n
--------------------------------------------
      ...  | 1 | 0 | ...
--------------------------------------------

c. rank n-2 reads staled rank n's array (no cache consistency ensurance), and write
   the element at position n-2 (in round 2).
         ↓       n
--------------------------------------------
   ... | 1 | 0 | 0 | ...
--------------------------------------------

d. rank n-2 overwrites rank n-1，so rank n may miss rank n-1's signal and wait forever.
             ↑   n
--------------------------------------------
   ... | 1 | 0 | 0 | ...
--------------------------------------------

To avoid this issue, separate elements must exist on different cachelines. See aclshmemi_sync_bit for detailed definition.

Additionly, instead of simply write a flag, each rank writes a 64-bit number into the array,
indicating how many times this team has performed barrier.

The temporal and spatial complexity of this implementation are O(logN) and O(N), respectively.

3. Futher development
  a. Hierarchical synchronization.
    Sync within the host first, then sync between host. May achieve better
    performance by utilizing low-latency in-host network better.

  b. Group dissemination.
    Group the ranks so that each rank could issue multiple signals and waits concurrently,
    instead of 1 signal and 1 wait as above.

  c. Optimize spatial complexity to O(logN).
*/

ACLSHMEM_DEVICE void aclshmemi_barrier_npu_v1(aclshmemx_team_t *team);

/** Group Dissemination Barrier.
 *
 *  An optimized version of aclshmemi_barrier_npu_v1, with temporal complexity reduced to O(log_{k}^{N}).
 */
ACLSHMEM_DEVICE void aclshmemi_barrier_npu_v2(aclshmemx_team_t *team);

/** Centralized Barrier (pull mode).
 *
 *  The temporal and spatial complexity of this implementation are O(N/K) and O(1), respectively.
 *  Performs better than Group Dissemination Barrier at small scale (eg. 8 ranks).
 */
ACLSHMEM_DEVICE void aclshmemi_barrier_npu_v3(aclshmemx_team_t *team);

/* Level 3: barrier between hosts, TO BE IMPLEMENTED. */
ACLSHMEM_DEVICE void aclshmemi_barrier_sys();

template<bool is_aiv_only = true>
ACLSHMEM_DEVICE void aclshmemi_barrier(aclshmem_team_t tid);

ACLSHMEM_DEVICE void aclshmemi_barrier_cross_host(aclshmemx_team_t *team);

ACLSHMEM_DEVICE void aclshmemi_handle(aclshmem_team_t tid);

ACLSHMEM_DEVICE void dcci_cacheline(__gm__ uint8_t * addr);

ACLSHMEM_DEVICE void dcci_cachelines(__gm__ uint8_t* addr, uint64_t length);

ACLSHMEM_DEVICE void dcci_entire_cache();

ACLSHMEM_DEVICE void dcci_atomic();

ACLSHMEM_DEVICE void dsb_all();

#endif