/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_DEVICE_SDMA_H
#define SHMEMI_DEVICE_SDMA_H

#include "kernel_operator.h"
#include "device/shmem_def.h"

struct sdma_config_t {
    uint64_t block_bytes;
    uint64_t per_core_bytes;
    uint32_t queue_num;
    uint32_t iter_num;
};

struct workspace_layout_t {
    __gm__ uint8_t* send_workspace;
    __gm__ uint8_t* recv_workspace;
    __gm__ uint8_t* remote_recv_workspace;
};

struct stars_channel_flag_info_t {
    uint32_t flag;
    uint32_t totalQueueNum;
    uint8_t reserved[56]; // 对齐到64字节
};

struct stars_channel_info_t {
    uint32_t sq_head;
    uint32_t sq_tail;
    uint64_t sq_base;        // SQ缓冲区基地址
    uint64_t sq_reg_base;    // SQ寄存器基地址
    uint32_t sq_depth;
    uint32_t sq_id;
    uint32_t cq_id;
    uint32_t logic_cq_id;
    uint64_t cqe_addr;
    uint32_t report_cqe_num;
    uint32_t stream_id;
    uint32_t dev_id;
    uint8_t reserved[4]; // 对齐到64字节
};

struct stars_sdma_sqe_t {
    uint8_t type : 6;
    uint16_t res1 : 10;
    uint16_t blockDim;
    uint16_t rtStreamId;
    uint16_t taskId;
    /**** 8 bytpes ****/

    uint32_t res3;

    uint16_t res4;
    uint8_t kernel_credit;
    uint8_t ptr_mode : 1;
    uint8_t res5 : 7;
    /**** 16 bytpes ****/

    uint32_t opcode : 8;
    uint32_t ie2 : 1;
    uint32_t sssv : 1;
    uint32_t dssv : 1;
    uint32_t sns : 1;
    uint32_t dns : 1;
    uint32_t qos : 4;
    uint32_t sro : 1;
    uint32_t dro : 1;
    uint32_t partid : 8;
    uint32_t mpam : 1;
    uint32_t res6 : 4;
    /**** 20 bytpes ****/

    uint16_t src_streamid;
    uint16_t src_sub_streamid;
    /**** 24 bytpes ****/

    uint16_t dst_streamid;
    uint16_t dst_sub_streamid;

    uint32_t length;
    /**** 32 bytpes ****/

    uint32_t srcAddrLow;
    uint32_t srcAddrHigh;
    uint32_t dstAddrLow;
    uint32_t dstAddrHigh;
    /**** 48 bytpes ****/

    uint8_t linkType;
    uint8_t resvered[3];
    uint32_t reslast[3];
    /**** 64 bytpes ****/
};

ACLSHMEM_DEVICE void aclshmemi_sdma_submit_data_sqes(__gm__ stars_channel_info_t *batch_write_channel_info,
                                                     __gm__ uint8_t *send_buffer, __gm__ uint8_t *recv_buffer,
                                                     const sdma_config_t &config, uint32_t *sq_tail);

ACLSHMEM_DEVICE void aclshmemi_sdma_submit_flag_sqes(__gm__ stars_channel_info_t *batch_write_channel_info,
                                                     const workspace_layout_t &layout, const sdma_config_t &config,
                                                     uint32_t *sq_tail);

ACLSHMEM_DEVICE void aclshmemi_sdma_poll_for_completion(AscendC::LocalTensor<uint32_t> &tmp_local, uint32_t sync_id);

/**
 * @brief AIV direct STARS helper function for post send, prepare SQE and ring doorbell.
 *
 * @param recv_buffer               [in] Destination buffer for received data
 * @param send_buffer               [in] Source buffer for data to be sent
 * @param message_len               [in] message length in Bytes
 * @param tmp_local                 [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                   [in] ID used to sync pipeline.
 */
ACLSHMEM_DEVICE void aclshmemi_sdma_post_send(__gm__ uint8_t *recv_buffer, __gm__ uint8_t *send_buffer,
                                              uint64_t message_len, AscendC::LocalTensor<uint32_t> &tmp_local,
                                              uint32_t sync_id);

/**
 * @brief SDMA Quiet function. This synchronous function ensures all previous SDMA SQEs are completed.
 *
 * @param buf                       [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                   [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemi_sdma_quiet(AscendC::LocalTensor<T> &buf, uint32_t sync_id);

/**
 * @brief SDMA Quiet function. This synchronous function ensures all previous SDMA SQEs are completed.
 *
 * @param buf                       [in] Pointer on local UB.
 * @param ub_size                   [in] The size of temp Buffer on UB. (In Bytes)
 * @param sync_id                   [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemi_sdma_quiet(__ubuf__ T *buf, uint32_t ub_size, uint32_t sync_id);

#endif
