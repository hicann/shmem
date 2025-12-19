/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MF_HYBM_SPACE_ALLOCATOR_H
#define MF_HYBM_SPACE_ALLOCATOR_H

#include <cstdint>
#include <memory>

namespace ock::mf {
class AllocatedElement {
public:
    AllocatedElement() noexcept : startAddress{nullptr}, size{0} {}

    explicit AllocatedElement(uint8_t *p, uint64_t s) noexcept : startAddress{p}, size{s} {}

    uint8_t *Address() noexcept
    {
        return startAddress;
    }

    const uint8_t *Address() const noexcept
    {
        return startAddress;
    }

    uint64_t Size() const noexcept
    {
        return size;
    }

    void Reset() noexcept
    {
        startAddress = nullptr;
        size = 0;
    }

private:
    uint8_t *startAddress;
    uint64_t size;
};

class SpaceAllocator {
public:
    virtual ~SpaceAllocator() = default;

public:
    virtual bool CanAllocate(uint64_t size) const noexcept = 0;
    virtual AllocatedElement Allocate(uint64_t size) noexcept = 0;
    virtual bool Release(const AllocatedElement &element) noexcept = 0;
};

using SpaceAllocatorPtr = std::shared_ptr<SpaceAllocator>;
}

#endif  // MF_HYBM_SPACE_ALLOCATOR_H
