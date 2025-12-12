/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <acl/acl.h>
#include <netinet/in.h>
#include <errno.h>
#include <unistd.h>
#include <gtest/gtest.h>
#include "aclshmemi_host_common.h"
#include "host/aclshmem_host_def.h"
#include "unittest_main_test.h"

namespace shm {
extern shmem_init_attr_t g_attr;
constexpr uint32_t timeout = 50;
void logger_test_example(int level, const char* msg)
{
    // do print here
}
}

void test_shmem_init(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = SHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    shmem_set_conf_store_tls(false, nullptr, 0);
    shmem_init_attr_t *attributes;
    shmem_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);

    status = shmem_init_attr(SHMEMX_INIT_WITH_DEFAULT, attributes);
    EXPECT_EQ(status, SHMEM_SUCCESS);
    EXPECT_EQ(g_state.mype, rank_id);
    EXPECT_EQ(g_state.npes, n_ranks);
    EXPECT_NE(g_state.heap_base, nullptr);
    EXPECT_NE(g_state.host_p2p_heap_base[rank_id], nullptr);
    EXPECT_EQ(g_state.heap_size, local_mem_size + SHMEM_EXTRA_SIZE);
    EXPECT_NE(g_state.team_pools[0], nullptr);
    status = shmem_init_status();
    EXPECT_EQ(status, SHMEM_STATUS_IS_INITIALIZED);
    status = shmem_finalize();
    EXPECT_EQ(status, SHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()) {
        exit(1);
    }
}

void test_shmem_init_invalid_rank_id(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    int erank_id = -1;
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = SHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    shmem_set_conf_store_tls(false, nullptr, 0);

    shmem_init_attr_t *attributes;
    shmem_set_attr(erank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);

    status = shmem_init_attr(SHMEMX_INIT_WITH_DEFAULT, attributes);

    EXPECT_EQ(status, SHMEM_INVALID_VALUE);
    status = shmem_init_status();
    EXPECT_EQ(status, SHMEM_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()) {
        exit(1);
    }
}

void test_shmem_init_invalid_n_ranks(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    int en_ranks = SHMEM_MAX_RANKS + 1;
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = SHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    shmem_set_conf_store_tls(false, nullptr, 0);
    shmemx_uniqueid_t uid;

    shmem_init_attr_t *attributes;
    shmem_set_attr(rank_id, en_ranks, local_mem_size, test_global_ipport, &attributes);

    status = shmem_init_attr(SHMEMX_INIT_WITH_DEFAULT, attributes);

    EXPECT_EQ(status, SHMEM_INVALID_VALUE);
    status = shmem_init_status();
    EXPECT_EQ(status, SHMEM_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()) {
        exit(1);
    }
}

void test_shmem_init_rank_id_over_size(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = SHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    shmem_set_conf_store_tls(false, nullptr, 0);

    shmem_init_attr_t *attributes;
    shmem_set_attr(rank_id + n_ranks, n_ranks, local_mem_size, test_global_ipport, &attributes);

    status = shmem_init_attr(SHMEMX_INIT_WITH_DEFAULT, attributes);
    EXPECT_EQ(status, SHMEM_INVALID_PARAM);
    status = shmem_init_status();
    EXPECT_EQ(status, SHMEM_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()) {
        exit(1);
    }
}

void test_shmem_init_zero_mem(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    // local_mem_size = 0
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = SHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    shmem_set_conf_store_tls(false, nullptr, 0);
    shmem_init_attr_t *attributes;
    shmem_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);

    status = shmem_init_attr(SHMEMX_INIT_WITH_DEFAULT, attributes);
    EXPECT_EQ(status, SHMEM_INVALID_VALUE);
    status = shmem_init_status();
    EXPECT_EQ(status, SHMEM_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()) {
        exit(1);
    }
}

TEST(TestInitAPI, TestShmemInit)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_shmem_init, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemInitErrorInvalidRankId)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_shmem_init_invalid_rank_id, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemInitErrorInvalidNRanks)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_shmem_init_invalid_n_ranks, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemInitErrorRankIdOversize)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_shmem_init_rank_id_over_size, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemInitErrorZeroMem)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 0;
    test_mutil_task(test_shmem_init_zero_mem, local_mem_size, process_count);
}

TEST(TestInitAPI, TestInfoGetVersion)
{
    int major = 0;
    int minor = 0;
    shmem_info_get_version(&major, &minor);
    EXPECT_EQ(major, SHMEM_MAJOR_VERSION);
    EXPECT_EQ(minor, SHMEM_MINOR_VERSION);
}

TEST(TestInitAPI, TestInfoGetVersionNull)
{
    int major = 0;
    shmem_info_get_version(&major, nullptr);
    EXPECT_EQ(major, 0);
}

TEST(TestInitAPI, TestInfoGetName)
{
    char name[256];
    shmem_info_get_name(name);
    EXPECT_TRUE(strlen(name) > 0);

    std::string expect = "SHMEM v" + std::to_string(SHMEM_VENDOR_MAJOR_VER) + "."
        + std::to_string(SHMEM_VENDOR_MINOR_VER) + "." + std::to_string(SHMEM_VENDOR_PATCH_VER).c_str();
    for (size_t i = 0; i < expect.length(); i++) {
        EXPECT_EQ(expect[i], name[i]);
    }
}

TEST(TestInitAPI, TestInfoGetNameNull)
{
    char *input = nullptr;
    shmem_info_get_name(input);
    EXPECT_EQ(input, nullptr);
}

TEST(TestInitAPI, TestShmemSetLogLevel)
{
    auto ret = shmem_set_log_level(shm::DEBUG_LEVEL);
    EXPECT_EQ(ret, 0);

    char* original_log_level = NULL;
    const char* env_val = getenv("SHMEM_LOG_LEVEL");
    if (env_val != NULL) {
        original_log_level = strdup(env_val);
    }

    setenv("SHMEM_LOG_LEVEL", "DEBUG", 1);
    EXPECT_EQ(shmem_set_log_level(-1), 0);

    setenv("SHMEM_LOG_LEVEL", "INFO", 1);
    EXPECT_EQ(shmem_set_log_level(-1), 0);

    setenv("SHMEM_LOG_LEVEL", "WARN", 1);
    EXPECT_EQ(shmem_set_log_level(-1), 0);

    setenv("SHMEM_LOG_LEVEL", "ERROR", 1);
    EXPECT_EQ(shmem_set_log_level(-1), 0);

    setenv("SHMEM_LOG_LEVEL", "FATAL", 1);
    EXPECT_EQ(shmem_set_log_level(-1), 0);

    unsetenv("SHMEM_LOG_LEVEL");
    if (original_log_level != NULL) {
        setenv("SHMEM_LOG_LEVEL", original_log_level, 1);
        free(original_log_level);
        original_log_level = NULL;
    }
}