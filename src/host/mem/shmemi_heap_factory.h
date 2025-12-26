/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HOST_MEM_SHMEMI_HEAP_FACTORY_H
#define HOST_MEM_SHMEMI_HEAP_FACTORY_H

#include <vector>
#include <map>
#include <mutex>
#include <memory>
#include "shmemi_heap_base.h"

// Type definition for heap creation function pointer
typedef std::unique_ptr<aclshmem_symmetric_heap_base> (*heap_create_func)(int pe_id, int pe_size, int dev_id, aclshmem_mem_type_t mem_type);

/**
 * @class aclshmem_symmetric_heap_factory
 * @brief Factory class for creating symmetric heap instances using registry pattern
 * 
 * This class implements a singleton factory that allows different heap implementations
 * to register themselves and be dynamically created based on memory type.
 */
class aclshmem_symmetric_heap_factory {
public:
    /**
     * @brief Get the singleton instance of the factory
     * @return Reference to the factory instance
     */
    static aclshmem_symmetric_heap_factory& get_instance() {
        static aclshmem_symmetric_heap_factory instance;
        return instance;
    }

    /**
     * @brief Register a heap creator function for a specific memory type
     * @param mem_type Memory type for which the creator is registered
     * @param func Creator function pointer
     * @return true if registration succeeded, false if already registered
     */
    bool register_heap_creator(aclshmem_mem_type_t mem_type, heap_create_func func) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (creators_.find(mem_type) != creators_.end()) {
            return false; // Creator for this type already exists
        }
        creators_[mem_type] = func;
        return true;
    }

    /**
     * @brief Create a heap instance based on memory type
     * @param mem_type Type of memory for the heap
     * @param pe_id Processing element ID
     * @param pe_size Number of processing elements
     * @param dev_id Device ID
     * @return Smart pointer to created heap instance, or empty if failed
     */
    std::unique_ptr<aclshmem_symmetric_heap_base> create_heap(aclshmem_mem_type_t mem_type, int pe_id, int pe_size, int dev_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = creators_.find(mem_type);
        if (it == creators_.end()) {
            return nullptr; // No creator registered for this type
        }
        return it->second(pe_id, pe_size, dev_id, mem_type);
    }

private:
    aclshmem_symmetric_heap_factory() = default;
    aclshmem_symmetric_heap_factory(const aclshmem_symmetric_heap_factory&) = delete;
    aclshmem_symmetric_heap_factory& operator=(const aclshmem_symmetric_heap_factory&) = delete;

    // Map storing creator functions for different memory types
    std::map<aclshmem_mem_type_t, heap_create_func> creators_;
    std::mutex mutex_;
};

/**
 * @class heap_creator_register
 * @brief Template class for automatic heap creator registration
 */
template <typename HeapType, aclshmem_mem_type_t MemType>
struct heap_creator_register {
private:
    /**
     * @brief Creator function implementation
     * @param pe_id Processing element ID
     * @param pe_size Number of processing elements
     * @param dev_id Device ID
     * @param type Memory type
     * @return Smart pointer to created heap instance
     */
    static std::unique_ptr<aclshmem_symmetric_heap_base> create(int pe_id, int pe_size, int dev_id, aclshmem_mem_type_t type) {
        return std::unique_ptr<aclshmem_symmetric_heap_base>(new HeapType(pe_id, pe_size, dev_id, type));
    }
    
    /**
     * @brief Static registration function
     * @return true if registration succeeded, false otherwise
     */
    static bool register_creator() {
        return aclshmem_symmetric_heap_factory::get_instance().register_heap_creator(MemType, create);
    }
    
    // Static variable that triggers registration during initialization
    static const bool registered_;
};

// Initialize the static registration flag
template <typename HeapType, aclshmem_mem_type_t MemType>
const bool heap_creator_register<HeapType, MemType>::registered_ = 
    heap_creator_register<HeapType, MemType>::register_creator();

/**
 * @def REGISTER_HEAP_CREATOR
 * @brief Macro for convenient registration of heap implementation classes
 * @param mem_type Memory type to register for
 * @param class_name Name of the heap implementation class
 */
#define REGISTER_HEAP_CREATOR(mem_type, class_name)  \
    template struct heap_creator_register<class_name, mem_type>

#endif  // HOST_MEM_SHMEMI_HEAP_FACTORY_H