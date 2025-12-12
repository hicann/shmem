/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_HOST_DEF_H
#define SHMEM_HOST_DEF_H
#include <climits>
#include <stdint.h>

#include <sys/socket.h>
#include <netinet/in.h>

#include "host_device/common_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
* @brief Standard RMA Types and Names valid on Host
*
* |NAME       | TYPE      |
* |-----------|-----------|
* |float      | float     |
* |double     | double    |
* |int8       | int8      |
* |int16      | int16     |
* |int32      | int32     |
* |int64      | int64     |
* |uint8      | uint8     |
* |uint16     | uint16    |
* |uint32     | uint32    |
* |uint64     | uint64    |
* |char       | char      |
*/
#define SHMEM_TYPE_FUNC(FUNC) \
    FUNC(float, float);       \
    FUNC(double, double);     \
    FUNC(int8, int8_t);       \
    FUNC(int16, int16_t);     \
    FUNC(int32, int32_t);     \
    FUNC(int64, int64_t);     \
    FUNC(uint8, uint8_t);     \
    FUNC(uint16, uint16_t);   \
    FUNC(uint32, uint32_t);   \
    FUNC(uint64, uint64_t);   \
    FUNC(char, char)
/**
 * @defgroup group_macros Macros
 * @{
*/
/// \def SHMEM_HOST_API
/// \brief A macro that identifies a function on the host side.
#define SHMEM_HOST_API __attribute__((visibility("default")))

/// \def SHMEM_XXX_VERSION
/// \brief macros that define current version info
#define SHMEM_MAJOR_VERSION 1
#define SHMEM_MINOR_VERSION 1
#define SHMEM_MAX_NAME_LEN 256
#define SHMEM_VENDOR_MAJOR_VER 1
#define SHMEM_VENDOR_MINOR_VER 1
#define SHMEM_VENDOR_PATCH_VER 1
#define SHMEM_MAX_IP_PORT_LEN 64
/**@} */  // end of group_macros

/**
 * @defgroup group_enums Enumerations
 * @{
*/

/**
 * @brief Error code for the SHMEM library.
*/
enum shmem_error_code_t : int {
    SHMEM_SUCCESS = 0,         ///< Task execution was successful.
    SHMEM_INVALID_PARAM = -1,  ///< There is a problem with the parameters.
    SHMEM_INVALID_VALUE = -2,  ///< There is a problem with the range of the value of the parameter.
    SHMEM_SMEM_ERROR = -3,     ///< There is a problem with SMEM.
    SHMEM_INNER_ERROR = -4,    ///< This is a problem caused by an internal error.
    SHMEM_NOT_INITED = -5,     ///< This is a problem caused by an uninitialization.
    SHMEM_BOOTSTRAP_ERROR = -6,///< This is a problem with BOOTSTRAP.
    SHMEM_TIMEOUT_ERROR = -7,  ///< This is a problem caused by TIMEOUT.
};

/**
 * @brief init flags
*/
enum shmemx_bootstrap_t : int {
    SHMEMX_INIT_WITH_UNIQUEID = 1,
    SHMEMX_INIT_WITH_MPI = 1 << 1,
    SHMEMX_INIT_WITH_DEFAULT = 1 << 2,
};

/**
 * @brief The state of the SHMEM library initialization.
*/
enum shmem_init_status_t {
    SHMEM_STATUS_NOT_INITIALIZED = 0,  ///< Uninitialized.
    SHMEM_STATUS_SHM_CREATED,          ///< Shared memory heap creation is complete.
    SHMEM_STATUS_IS_INITIALIZED,       ///< Initialization is complete.
    SHMEM_STATUS_INVALID = INT_MAX,    ///< Invalid status code.
};

/**
 * @brief Different transports supported by SHMEM library.
*/
enum shmem_transport_t : uint8_t {
    SHMEM_TRANSPORT_MTE = 1 << 0,    ///< MTE Transport.
    SHMEM_TRANSPORT_ROCE = 1 << 1,   ///< RDMA Transport (RoCE).
};

/**@} */  // end of group_enums

/**
 * @defgroup group_structs Structs
 * @{
*/

/**
 * @struct shmem_init_optional_attr_t
 * @brief Optional parameter for the attributes used for initialization.
 *
 * - int version: version
 * - data_op_engine_type_t data_op_engine_type: data_op_engine_type
 * - uint32_t shm_init_timeout: shm_init_timeout
 * - uint32_t shm_create_timeout: shm_create_timeout
 * - uint32_t control_operation_timeout: control_operation_timeout
 * - int32_t sockFd: sock_fd for apply port in advance
*/
typedef struct {
    int version;
    data_op_engine_type_t data_op_engine_type;
    uint32_t shm_init_timeout;
    uint32_t shm_create_timeout;
    uint32_t control_operation_timeout;
    int32_t sockFd;
} shmem_init_optional_attr_t;

/**
 * @struct shmem_init_attr_t
 * @brief Mandatory parameter for attributes used for initialization.
 *
 * - int my_rank: The rank of the current process.
 * - int n_ranks: The total rank number of all processes.
 * - char ip_port[SHMEM_MAX_IP_PORT_LEN]: The ip and port of the communication server. The port must not conflict
 *   with other modules and processes.
 * - uint64_t local_mem_size: The size of shared memory currently occupied by current rank.
 * - shmem_init_optional_attr_t option_attr: Optional Parameters.
*/
typedef struct {
    int my_rank;
    int n_ranks;
    char ip_port[SHMEM_MAX_IP_PORT_LEN];
    uint64_t local_mem_size;
    shmem_init_optional_attr_t option_attr;
    void *comm_args;
} shmem_init_attr_t;

/**
 * @brief Callback function of private key password decryptor, see shmem_set_config_store_tls_key
 *
 * @param cipherText       [in] the encrypted text(private password)
 * @param cipherTextLen    [in] the length of encrypted text
 * @param plainText        [out] the decrypted text(private password)
 * @param plainTextLen     [out] the length of plainText
 */
typedef int (*shmem_decrypt_handler)(const char *cipherText, size_t cipherTextLen, char *plainText,
                                     size_t &plainTextLen);

typedef enum {
    ADDR_IPv4,
    ADDR_IPv6
} addr_type_t;

// shmem unique id
typedef struct {
    union {
        struct sockaddr sa;
        struct sockaddr_in addr4;  // IPv4地址（含端口）
        struct sockaddr_in6 addr6; // IPv6地址（含端口）
    } addr;
    addr_type_t type;
} sockaddr_t;

typedef struct {
    int32_t version;
    int32_t inner_sockFd;          // for mf backend              
    sockaddr_t addr;              // 动态传入的地址（含端口）
    uint64_t magic;
    int rank;
    int nranks;
} shmemx_bootstrap_uid_state_t;

constexpr uint16_t SHMEM_UNIQUE_ID_INNER_LEN = 124;

typedef struct {
    int32_t version;
    char internal[SHMEM_UNIQUE_ID_INNER_LEN];
} shmemx_uniqueid_t;

constexpr int32_t SHMEM_UNIQUEID_VERSION = (1 << 16) + sizeof(shmemx_uniqueid_t);

#define SHMEM_UNIQUEID_INITIALIZER                      \
    {                                                   \
        SHMEM_UNIQUEID_VERSION,                         \
        {                                               \
            0                                           \
        }                                               \
    }                                                   \

#ifdef __cplusplus
}
#endif

#endif