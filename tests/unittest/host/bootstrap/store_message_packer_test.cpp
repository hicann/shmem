/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <gtest/gtest.h>

#include "bootstrap/config_store/store_message_packer.h"

namespace shm {
namespace store {
namespace {
template <typename T>
void WritePod(std::vector<uint8_t>& buffer, const uint64_t offset, const T& value)
{
    std::memcpy(buffer.data() + offset, &value, sizeof(value));
}

TEST(SmemMessagePackerTest, RejectsTruncatedKeyLengthField)
{
    // Fixed header: totalSize, userDef, keyCount, and valueCount are 8 bytes
    // each; MessageType is an int16_t. Therefore baseSize is 34 bytes.
    constexpr uint64_t baseSize = 4U * sizeof(uint64_t) + sizeof(MessageType);
    std::vector<uint8_t> buffer(baseSize, 0U);
    const uint64_t totalSize = baseSize;
    const int64_t userDef = 0;
    const MessageType messageType = MessageType::GET;
    // The first keySize occupies the trailing valueCount-sized slot in the
    // fixed header. The second keySize starts immediately beyond totalSize.
    const uint64_t keyCount = 2;

    uint64_t offset = 0;
    WritePod(buffer, offset, totalSize);
    offset += sizeof(totalSize);
    WritePod(buffer, offset, userDef);
    offset += sizeof(userDef);
    WritePod(buffer, offset, messageType);
    offset += sizeof(messageType);
    WritePod(buffer, offset, keyCount);

    SmemMessage message;
    EXPECT_EQ(SmemMessagePacker::Unpack(buffer.data(), buffer.size(), message), -1);
}
} // namespace
} // namespace store
} // namespace shm
