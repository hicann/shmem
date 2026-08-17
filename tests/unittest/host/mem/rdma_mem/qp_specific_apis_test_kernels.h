/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QP_SPECIFIC_APIS_TEST_KERNELS_H
#define QP_SPECIFIC_APIS_TEST_KERNELS_H

#include <cstdint>

void test_rdma_roce_qp_put_nbi_raw_do(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);
void test_rdma_roce_qp_get_nbi_raw_do(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);
void test_rdma_roce_qp_put_nbi_tensor_do(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);
void test_rdma_roce_qp_get_nbi_tensor_do(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);

#endif // QP_SPECIFIC_APIS_TEST_KERNELS_H
