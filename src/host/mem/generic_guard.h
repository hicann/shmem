/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _HOST_MEM_GENERIC_GUARD_H_
#define _HOST_MEM_GENERIC_GUARD_H_
namespace shmem {
// Generic resource guard class with RAII
// T: Resource type
// D: Deleter type (function, functor, lambda, etc.)
template <typename T, typename D>
class generic_guard {
public:
    // Constructor
    generic_guard(T resource, D deleter) 
        : resource_(resource), deleter_(std::move(deleter)), owns_resource_(true) {}
    
    // Destructor - automatically releases resource
    ~generic_guard() {
        release_resource();
    }
    
    // Disable copy operations
    generic_guard(const generic_guard&) = delete;
    generic_guard& operator=(const generic_guard&) = delete;
    
    // Enable move operations
    generic_guard(generic_guard&& other) noexcept 
        : resource_(other.resource_), deleter_(std::move(other.deleter_)), owns_resource_(other.owns_resource_) {
        other.owns_resource_ = false;
    }
    
    generic_guard& operator=(generic_guard&& other) noexcept {
        if (this != &other) {
            release_resource();
            resource_ = other.resource_;
            deleter_ = std::move(other.deleter_);
            owns_resource_ = other.owns_resource_;
            other.owns_resource_ = false;
        }
        return *this;
    }
    
    // Manually release the resource
    void release() {
        release_resource();
    }
    
    T release_ownership()
    {
        owns_resource_ = false;
        return resource_;
    }
    // Get the resource
    T get() const {
        return resource_;
    }
    
    // Check if the guard owns the resource
    bool owns() const {
        return owns_resource_;
    }
    
    // Reset with a new resource
    void reset(T new_resource) {
        release_resource();
        resource_ = new_resource;
        owns_resource_ = true;
    }
    
private:
    // Release the resource if owned
    void release_resource() {
        if (owns_resource_) {
            deleter_(resource_);
            owns_resource_ = false;
        }
    }
    
    T resource_;
    D deleter_;
    bool owns_resource_;
};

// Helper function to create a generic_guard
// Usage: auto guard = make_guard(resource, deleter);
template <typename T, typename D>
auto make_guard(T resource, D deleter) {
    return generic_guard<T, D>(resource, std::move(deleter));
}
}
#endif