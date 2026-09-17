/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <limits>

#include "device_udma_transport_manager.h"
#include "shmemi_host_common.h"

namespace shm {
namespace transport {
namespace device {

class UdmaEndpointExchangeTest : public testing::Test {
protected:
    using Descriptor = UdmaTransportManager::ExchangedEndpointDesc;
    using Exchange = UdmaTransportManager::EndpointExchange;

    void SetUp() override
    {
        saved_handle_ = g_boot_handle;
        g_boot_handle.npes = 3;
        g_boot_handle.allgather = AllGather;
        lengths_.clear();
        fail_call_ = 0;
    }

    void TearDown() override { g_boot_handle = saved_handle_; }

    static int AllGather(const void* send, void* recv, int len, aclshmemi_bootstrap_handle_t* handle)
    {
        lengths_.push_back(len);
        if (lengths_.size() == fail_call_) {
            return ACLSHMEM_BOOTSTRAP_ERROR;
        }
        const size_t count = static_cast<size_t>(len) / sizeof(Descriptor);
        const auto* input = static_cast<const Descriptor*>(send);
        auto* output = static_cast<Descriptor*>(recv);
        for (int rank = 0; rank < handle->npes; ++rank) {
            for (size_t idx = 0; idx < count; ++idx) {
                output[rank * count + idx] = input[idx];
                output[rank * count + idx].peer_rank = static_cast<uint32_t>(rank);
            }
        }
        return ACLSHMEM_SUCCESS;
    }

    static Result Gather(uint32_t chunk_count, Exchange& exchange)
    {
        exchange.max_count = 5;
        std::vector<Descriptor> local(exchange.max_count);
        for (uint32_t idx = 0; idx < exchange.max_count; ++idx) {
            local[idx].eid_index = idx;
            local[idx].valid = idx < 4 ? 1 : 0;
            local[idx].listen_port = static_cast<uint16_t>(100 + idx);
        }
        return UdmaTransportManager::GatherEndpointChunks(local, 3, chunk_count, exchange);
    }

    static uint32_t ChunkCount(uint32_t ranks) { return UdmaTransportManager::GetEndpointChunkCount(ranks); }

    static bool GetSize(uint32_t ranks, uint32_t endpoints, size_t& count)
    {
        return UdmaTransportManager::GetEndpointExchangeSize(ranks, endpoints, count);
    }

    static uint32_t MaxEndpoints()
    {
        return std::numeric_limits<int>::max() / sizeof(UdmaTransportManager::ExchangedEndpointDesc);
    }

    aclshmemi_bootstrap_handle_t saved_handle_{};
    inline static std::vector<int> lengths_;
    inline static size_t fail_call_{0};
};

TEST_F(UdmaEndpointExchangeTest, AcceptsBoundaryAndRejectsNextEndpoint)
{
    constexpr uint32_t ranks = 1;
    const uint32_t endpoints = MaxEndpoints();
    size_t count = 0;
    ASSERT_TRUE(GetSize(ranks, endpoints, count));
    EXPECT_EQ(count, static_cast<size_t>(ranks) * endpoints);
    EXPECT_FALSE(GetSize(ranks, endpoints + 1, count));
    EXPECT_EQ(count, 0U);
}

TEST_F(UdmaEndpointExchangeTest, RejectsEmptyAndOverflowingDimensions)
{
    size_t count = 1;
    EXPECT_FALSE(GetSize(0, 1, count));
    EXPECT_EQ(count, 0U);
    EXPECT_FALSE(GetSize(1, 0, count));
    EXPECT_FALSE(GetSize(std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max(), count));
    EXPECT_EQ(count, 0U);
}

TEST_F(UdmaEndpointExchangeTest, AcceptsDescriptorMatricesAboveEightMiB)
{
    size_t count = 0;
    EXPECT_TRUE(GetSize(8, 7, count));
    EXPECT_EQ(count, 56U);
    EXPECT_TRUE(GetSize(256, 255, count));
    EXPECT_EQ(count, 256U * 255U);
    EXPECT_TRUE(GetSize(1024, 1023, count));
    EXPECT_EQ(count, 1024U * 1023U);
    EXPECT_TRUE(GetSize(1024, 8U * 1023U, count));
    EXPECT_EQ(count, 1024U * 8U * 1023U);
}

TEST_F(UdmaEndpointExchangeTest, SplitsLargeClusterMatricesWithoutAllocatingThem)
{
    for (uint32_t ranks : {3493U, 3494U, 4096U, 10000U, 16384U}) {
        size_t count = 0;
        ASSERT_TRUE(GetSize(ranks, ranks - 1, count));
        const uint32_t chunk = ChunkCount(ranks);
        ASSERT_GT(chunk, 0U);
        EXPECT_LT(chunk, ranks - 1);
        // Mirror config_store's int multiplication under UBSan.
        const int bytes = static_cast<int>(chunk * sizeof(Descriptor));
        const int receive_bytes = bytes * static_cast<int>(ranks);
        EXPECT_LE(receive_bytes, 32 * 1024 * 1024);
    }
    EXPECT_EQ(ChunkCount(0), 0U);
}

TEST_F(UdmaEndpointExchangeTest, PreservesRankStridePaddingAndTailAcrossChunks)
{
    Exchange exchange;
    ASSERT_EQ(Gather(2, exchange), ACLSHMEM_SUCCESS);
    EXPECT_EQ(lengths_, (std::vector<int>{2 * sizeof(Descriptor), 2 * sizeof(Descriptor), sizeof(Descriptor)}));
    ASSERT_EQ(exchange.descs.size(), 15U);
    for (uint32_t rank = 0; rank < 3; ++rank) {
        for (uint32_t idx = 0; idx < 5; ++idx) {
            const auto& desc = exchange.descs[rank * 5 + idx];
            EXPECT_EQ(desc.peer_rank, rank);
            EXPECT_EQ(desc.eid_index, idx);
            EXPECT_EQ(desc.valid, idx < 4 ? 1U : 0U);
            EXPECT_EQ(desc.listen_port, 100U + idx);
        }
    }
}

TEST_F(UdmaEndpointExchangeTest, SingleChunkMatchesSplitExchange)
{
    Exchange split;
    ASSERT_EQ(Gather(2, split), ACLSHMEM_SUCCESS);
    lengths_.clear();
    Exchange single;
    ASSERT_EQ(Gather(5, single), ACLSHMEM_SUCCESS);
    EXPECT_EQ(lengths_.size(), 1U);
    ASSERT_EQ(single.descs.size(), split.descs.size());
    for (size_t idx = 0; idx < split.descs.size(); ++idx) {
        EXPECT_EQ(single.descs[idx].eid_index, split.descs[idx].eid_index);
        EXPECT_EQ(single.descs[idx].peer_rank, split.descs[idx].peer_rank);
        EXPECT_EQ(single.descs[idx].valid, split.descs[idx].valid);
    }
}

TEST_F(UdmaEndpointExchangeTest, StopsAtFailedChunk)
{
    fail_call_ = 2;
    Exchange exchange;
    EXPECT_EQ(Gather(2, exchange), ACLSHMEM_BOOTSTRAP_ERROR);
    EXPECT_EQ(lengths_.size(), 2U);
}

} // namespace device
} // namespace transport
} // namespace shm
