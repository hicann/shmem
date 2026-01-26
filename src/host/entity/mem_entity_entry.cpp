/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "shmemi_logger.h"
#include "mem_entity_inter.h"
#include "dl_api.h"
#include "dl_hal_api.h"
#include "dl_acl_api.h"
#include "dl_comm_def.h"
#include "devmm_svm_gva.h"
#include "devmm_cmd.h"
#include "hybm_ex_info_transfer.h"
#include "shmemi_file_util.h"
#include "mem_entity_factory.h"
#include "mem_entity_entry.h"

using namespace shm;

namespace {
static uint64_t g_baseAddr = 0ULL;
int64_t initialized = 0;
uint16_t initedDeviceId = 0;
int32_t initedLogicDeviceId = -1;

std::mutex initMutex;
}

int32_t HybmGetInitDeviceId()
{
    return static_cast<int32_t>(initedDeviceId);
}

bool HybmHasInited()
{
    return initialized > 0;
}

static inline int hybm_load_library()
{
    char *path = std::getenv("ASCEND_HOME_PATH");
    SHM_VALIDATE_RETURN(path != nullptr, "Environment ASCEND_HOME_PATH not set.", ACLSHMEM_INNER_ERROR);

    std::string libPath = std::string(path).append("/lib64");
    if (!shm::utils::FileUtil::Realpath(libPath) || !shm::utils::FileUtil::IsDir(libPath)) {
        SHM_LOG_ERROR("Environment ASCEND_HOME_PATH check failed.");
        return ACLSHMEM_INNER_ERROR;
    }
    auto ret = DlApi::LoadLibrary(libPath);
    SHM_LOG_ERROR_RETURN_IT_IF_NOT_OK(ret, "load library from path failed: " << ret);
    return 0;
}

HYBM_API int32_t hybm_init(uint16_t deviceId, uint64_t flags)
{
    std::unique_lock<std::mutex> lockGuard{initMutex};
    if (initialized > 0) {
        if (initedDeviceId != deviceId) {
            SHM_LOG_ERROR("this deviceId(" << deviceId << ") is not equal to the deviceId(" <<
                initedDeviceId << ") of other module!");
            return ACLSHMEM_INNER_ERROR;
        }

        /*
         * hybm_init will be accessed multiple times when bm/shm/trans init
         * incremental loading is required here.
         */
        SHM_LOG_ERROR_RETURN_IT_IF_NOT_OK(hybm_load_library(), "load library failed");

        initialized++;
        return 0;
    }

    SHM_LOG_ERROR_RETURN_IT_IF_NOT_OK(HalGvaPrecheck(), "the current version of ascend driver does not support mf!");

    SHM_LOG_ERROR_RETURN_IT_IF_NOT_OK(hybm_load_library(), "load library failed");

    auto ret = DlAclApi::RtGetLogicDevIdByUserDevId(deviceId, &initedLogicDeviceId);
    if (ret != 0 || initedLogicDeviceId < 0) {
        SHM_LOG_ERROR("fail to get logic device id " << deviceId << ", ret=" << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    SHM_LOG_INFO("success to get logic device user id=" << deviceId << ", logic deviceId = " << initedLogicDeviceId);
    ret = DlAclApi::AclrtSetDevice(deviceId);
    if (ret != ACLSHMEM_SUCCESS) {
        DlApi::CleanupLibrary();
        SHM_LOG_ERROR("set device id to be " << deviceId << " failed: " << ret);
        return ACLSHMEM_INNER_ERROR;
    }

    void *globalMemoryBase = nullptr;
    size_t allocSize = HYBM_DEVICE_INFO_SIZE;  // 申请meta空间
    drv::DevmmInitialize(initedLogicDeviceId, DlHalApi::GetFd());
    ret = drv::HalGvaReserveMemory((uint64_t *)&globalMemoryBase, allocSize, initedLogicDeviceId, flags);
    if (ret != 0) {
        DlApi::CleanupLibrary();
        SHM_LOG_ERROR("initialize mete memory with size: " << allocSize << ", flag: " << flags << " failed: " << ret);
        return ACLSHMEM_INNER_ERROR;
    }

    ret = drv::HalGvaAlloc(HYBM_DEVICE_META_ADDR, HYBM_DEVICE_INFO_SIZE, 0);
    if (ret != ACLSHMEM_SUCCESS) {
        DlApi::CleanupLibrary();
        int32_t hal_ret = drv::HalGvaUnreserveMemory((uint64_t)globalMemoryBase);
        SHM_LOG_ERROR("HalGvaAlloc hybm meta memory failed: " << ret << ", un-reserve memory " << hal_ret);
        return ACLSHMEM_MALLOC_FAILED;
    }

    initedDeviceId = deviceId;
    SHM_LOG_INFO("hybm_init end device id " << deviceId << ", logic device id " << initedLogicDeviceId);
    initialized = 1L;
    g_baseAddr = (uint64_t)globalMemoryBase;
    SHM_LOG_INFO("hybm init successfully.");
    return 0;
}

HYBM_API void hybm_uninit(void)
{
    std::unique_lock<std::mutex> lockGuard{initMutex};
    if (initialized <= 0L) {
        SHM_LOG_WARN("hybm not initialized.");
        return;
    }

    if (--initialized > 0L) {
        return;
    }

    drv::HalGvaFree(HYBM_DEVICE_META_ADDR, HYBM_DEVICE_INFO_SIZE);
    auto ret = drv::HalGvaUnreserveMemory(g_baseAddr);
    g_baseAddr = 0ULL;
    SHM_LOG_INFO("uninitialize GVA memory return: " << ret);
    initialized = 0;
}

HYBM_API hybm_entity_t hybm_create_entity(uint16_t id, const hybm_options *options, uint32_t flags)
{
    SHM_ASSERT_RETURN(HybmHasInited(), nullptr);

    auto &factory = MemEntityFactory::Instance();
    auto entity = factory.GetOrCreateEngine(id, flags);
    if (entity == nullptr) {
        SHM_LOG_ERROR("create entity failed.");
        return nullptr;
    }

    auto ret = entity->Initialize(options);
    if (ret != 0) {
        MemEntityFactory::Instance().RemoveEngine(entity.get());
        SHM_LOG_ERROR("initialize entity failed: " << ret);
        return nullptr;
    }

    return entity.get();
}

HYBM_API void hybm_destroy_entity(hybm_entity_t e, uint32_t flags)
{
    SHM_ASSERT_RET_VOID(e != nullptr);
    auto entity = MemEntityFactory::Instance().FindEngineByPtr(e);
    SHM_ASSERT_RET_VOID(entity != nullptr);
    entity->UnInitialize();
    MemEntityFactory::Instance().RemoveEngine(e);
}

HYBM_API int32_t hybm_reserve_mem_space(hybm_entity_t e, uint32_t flags, void **reservedMem)
{
    SHM_ASSERT_RETURN(e != nullptr, ACLSHMEM_INVALID_PARAM);
    auto entity = MemEntityFactory::Instance().FindEngineByPtr(e);
    SHM_ASSERT_RETURN(entity != nullptr, ACLSHMEM_INVALID_PARAM);
    SHM_ASSERT_RETURN(reservedMem != nullptr, ACLSHMEM_INVALID_PARAM);
    return entity->ReserveMemorySpace(reservedMem);
}

HYBM_API int32_t hybm_unreserve_mem_space(hybm_entity_t e, uint32_t flags, void *reservedMem)
{
    SHM_ASSERT_RETURN(e != nullptr, ACLSHMEM_INVALID_PARAM);
    auto entity = MemEntityFactory::Instance().FindEngineByPtr(e);
    SHM_ASSERT_RETURN(entity != nullptr, ACLSHMEM_INVALID_PARAM);
    return entity->UnReserveMemorySpace();
}

HYBM_API void *hybm_get_memory_ptr(hybm_entity_t e, hybm_mem_type mType)
{
    auto entity = static_cast<MemEntity *>(e);
    SHM_ASSERT_RETURN(entity != nullptr, nullptr);
    return entity->GetReservedMemoryPtr(mType);
}

HYBM_API hybm_mem_slice_t hybm_alloc_local_memory(hybm_entity_t e, hybm_mem_type mType, uint64_t size, uint32_t flags)
{
    SHM_ASSERT_RETURN(e != nullptr, nullptr);
    auto entity = MemEntityFactory::Instance().FindEngineByPtr(e);
    SHM_ASSERT_RETURN(entity != nullptr, nullptr);
    hybm_mem_slice_t slice;
    auto ret = entity->AllocLocalMemory(size, mType, flags, slice);
    if (ret != 0) {
        SHM_LOG_ERROR("allocate slice with size: " << size << " failed: " << ret);
        return nullptr;
    }

    return slice;
}

HYBM_API int32_t hybm_free_local_memory(hybm_entity_t e, hybm_mem_slice_t slice, uint32_t count, uint32_t flags)
{
    SHM_ASSERT_RETURN(e != nullptr, ACLSHMEM_INVALID_PARAM);
    auto entity = MemEntityFactory::Instance().FindEngineByPtr(e);
    SHM_ASSERT_RETURN(entity != nullptr, ACLSHMEM_INVALID_PARAM);
    SHM_ASSERT_RETURN(slice != nullptr, ACLSHMEM_INVALID_PARAM);
    return entity->FreeLocalMemory(slice, flags);
}

HYBM_API hybm_mem_slice_t hybm_register_local_memory(hybm_entity_t e, hybm_mem_type mType, const void *ptr,
                                                     uint64_t size, uint32_t flags)
{
    SHM_ASSERT_RETURN(e != nullptr, nullptr);
    auto entity = MemEntityFactory::Instance().FindEngineByPtr(e);
    SHM_ASSERT_RETURN(entity != nullptr, nullptr);

    hybm_mem_slice_t slice;
    auto ret = entity->RegisterLocalMemory(ptr, size, flags, slice);
    if (ret != 0) {
        SHM_LOG_ERROR("register slice with size: " << size << " failed: " << ret);
        return nullptr;
    }

    return slice;
}

HYBM_API int32_t hybm_export(hybm_entity_t e, hybm_mem_slice_t slice, uint32_t flags, hybm_exchange_info *exInfo)
{
    SHM_ASSERT_RETURN(e != nullptr, ACLSHMEM_INVALID_PARAM);
    auto entity = MemEntityFactory::Instance().FindEngineByPtr(e);
    SHM_ASSERT_RETURN(entity != nullptr, ACLSHMEM_INVALID_PARAM);
    SHM_ASSERT_RETURN(exInfo != nullptr, ACLSHMEM_INVALID_PARAM);

    ExchangeInfoWriter writer(exInfo);
    auto ret = entity->ExportExchangeInfo(slice, writer, flags);
    if (ret != 0) {
        SHM_LOG_ERROR("export slices failed: " << ret);
        return ret;
    }

    return ACLSHMEM_SUCCESS;
}

HYBM_API int32_t hybm_export_slice_size(hybm_entity_t e, size_t *size)
{
    SHM_ASSERT_RETURN(e != nullptr, ACLSHMEM_INVALID_PARAM);
    auto entity = MemEntityFactory::Instance().FindEngineByPtr(e);
    SHM_ASSERT_RETURN(entity != nullptr, ACLSHMEM_INVALID_PARAM);
    SHM_ASSERT_RETURN(size != nullptr, ACLSHMEM_INVALID_PARAM);

    auto ret = entity->GetExportSliceInfoSize(*size);
    return ret;
}

HYBM_API int32_t hybm_import(hybm_entity_t e, const hybm_exchange_info allExInfo[], uint32_t count, void *addresses[],
                             uint32_t flags)
{
    SHM_ASSERT_RETURN(e != nullptr, ACLSHMEM_INVALID_PARAM);
    auto entity = MemEntityFactory::Instance().FindEngineByPtr(e);
    SHM_ASSERT_RETURN(entity != nullptr, ACLSHMEM_INVALID_PARAM);
    SHM_ASSERT_RETURN(allExInfo != nullptr, ACLSHMEM_INVALID_PARAM);
    std::vector<ExchangeInfoReader> readers(count);
    for (auto i = 0U; i < count; i++) {
        readers[i].Reset(allExInfo + i);
    }

    return entity->ImportExchangeInfo(readers.data(), count, addresses, flags);
}

HYBM_API int32_t hybm_mmap(hybm_entity_t e, uint32_t flags)
{
    SHM_ASSERT_RETURN(e != nullptr, ACLSHMEM_INVALID_PARAM);
    auto entity = MemEntityFactory::Instance().FindEngineByPtr(e);
    SHM_ASSERT_RETURN(entity != nullptr, ACLSHMEM_INVALID_PARAM);
    return entity->Mmap();
}

HYBM_API int32_t hybm_entity_reach_types(hybm_entity_t e, uint32_t rank, hybm_data_op_type &reachTypes, uint32_t flags)
{
    auto entity = (MemEntity *)e;
    SHM_ASSERT_RETURN(entity != nullptr, ACLSHMEM_INVALID_PARAM);

    reachTypes = entity->CanReachDataOperators(rank);
    return ACLSHMEM_SUCCESS;
}

HYBM_API int32_t hybm_remove_imported(hybm_entity_t e, uint32_t rank, uint32_t flags)
{
    SHM_ASSERT_RETURN(e != nullptr, ACLSHMEM_INVALID_PARAM);
    auto entity = MemEntityFactory::Instance().FindEngineByPtr(e);
    SHM_ASSERT_RETURN(entity != nullptr, ACLSHMEM_INVALID_PARAM);

    std::vector<uint32_t> ranks = {rank};
    return entity->RemoveImported(ranks);
}

HYBM_API int32_t hybm_set_extra_context(hybm_entity_t e, const void *context, uint32_t size)
{
    SHM_ASSERT_RETURN(e != nullptr, ACLSHMEM_INVALID_PARAM);
    auto entity = MemEntityFactory::Instance().FindEngineByPtr(e);
    SHM_ASSERT_RETURN(entity != nullptr, ACLSHMEM_INVALID_PARAM);
    SHM_ASSERT_RETURN(context != nullptr, ACLSHMEM_INVALID_PARAM);
    auto ret = entity->SetExtraContext(context, size);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("SetExtraContext failed, ret: " << ret);
        return ret;
    }
    return ACLSHMEM_SUCCESS;
}

HYBM_API void hybm_unmap(hybm_entity_t e, uint32_t flags)
{
    SHM_ASSERT_RET_VOID(e != nullptr);
    auto entity = MemEntityFactory::Instance().FindEngineByPtr(e);
    SHM_ASSERT_RET_VOID(entity != nullptr);
    entity->Unmap();
}
