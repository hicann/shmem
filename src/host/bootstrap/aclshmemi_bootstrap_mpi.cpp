/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <assert.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "host/aclshmem_host_def.h"
#include "utils/aclshmemi_logger.h"
#include "utils/aclshmemi_host_types.h"
#include "init/bootstrap/aclshmemi_bootstrap.h"

typedef struct {
    MPI_Comm comm;
    int mpi_initialized;
} shmemi_bootstrap_mpi_state_t;
static shmemi_bootstrap_mpi_state_t shmemi_bootstrap_mpi_state = {MPI_COMM_NULL, 0};

static int shmemi_bootstrap_mpi_barrier(shmemi_bootstrap_handle_t *handle) {
    int status = MPI_SUCCESS;

    status = MPI_Barrier(shmemi_bootstrap_mpi_state.comm);
    SHMEM_CHECK_RET(status);

    return status;
}

static int shmemi_bootstrap_mpi_allgather(const void *sendbuf, void *recvbuf, int length,
                                   shmemi_bootstrap_handle_t *handle) {
    int status = MPI_SUCCESS;

    status = MPI_Allgather(sendbuf, length, MPI_BYTE, recvbuf, length, MPI_BYTE, shmemi_bootstrap_mpi_state.comm);
    SHMEM_CHECK_RET(status);

    return status;
}

static int shmemi_bootstrap_mpi_alltoall(const void *sendbuf, void *recvbuf, int length,
                                  shmemi_bootstrap_handle_t *handle) {
    int status = MPI_SUCCESS;

    status = MPI_Alltoall(sendbuf, length, MPI_BYTE, recvbuf, length, MPI_BYTE, shmemi_bootstrap_mpi_state.comm);
    SHMEM_CHECK_RET(status);

    return status;
}

static void shmemi_bootstrap_mpi_global_exit(int status) {
    int rc = MPI_SUCCESS;

    rc = MPI_Abort(shmemi_bootstrap_mpi_state.comm, status);
    if (rc != MPI_SUCCESS) {
        exit(1);
    }
}

static int shmemi_bootstrap_mpi_finalize(shmemi_bootstrap_handle_t *handle) {
    int status = MPI_SUCCESS, finalized;

    status = MPI_Finalized(&finalized);
    SHMEM_CHECK_RET(status);

    if (finalized) {
        if (shmemi_bootstrap_mpi_state.mpi_initialized) {
            status = SHMEM_INNER_ERROR;
        } else {
            status = 0;
        }

        return status;
    }

    if (!finalized && shmemi_bootstrap_mpi_state.mpi_initialized) {
        status = MPI_Comm_free(&shmemi_bootstrap_mpi_state.comm);
        SHMEM_CHECK_RET(status);
    }

    if (shmemi_bootstrap_mpi_state.mpi_initialized) MPI_Finalize();

    return status;
}

int shmemi_bootstrap_plugin_init(void *mpi_comm, shmemi_bootstrap_handle_t *handle) {
    int status = MPI_SUCCESS, initialized = 0, finalized = 0;
    MPI_Comm src_comm;
    if (NULL == mpi_comm)
        src_comm = MPI_COMM_WORLD;
    else
        src_comm = *((MPI_Comm *)mpi_comm);
    status = MPI_Initialized(&initialized);
    SHMEM_CHECK_RET(status);
    status = MPI_Finalized(&finalized);
    SHMEM_CHECK_RET(status);
    if (!initialized && !finalized) {
        MPI_Init(NULL, NULL);
        shmemi_bootstrap_mpi_state.mpi_initialized = 1;

        if (src_comm != MPI_COMM_WORLD && src_comm != MPI_COMM_SELF) {
            status = SHMEM_INNER_ERROR;
            if (shmemi_bootstrap_mpi_state.mpi_initialized) {
                MPI_Finalize();
                shmemi_bootstrap_mpi_state.mpi_initialized = 0;
            } 
        }
    } else if (finalized) {
        status = SHMEM_INNER_ERROR;
        if (shmemi_bootstrap_mpi_state.mpi_initialized) {
            MPI_Finalize();
            shmemi_bootstrap_mpi_state.mpi_initialized = 0;
        }
    }
    status = MPI_Comm_dup(src_comm, &shmemi_bootstrap_mpi_state.comm);
    SHMEM_CHECK_RET(status);
    status = MPI_Comm_rank(shmemi_bootstrap_mpi_state.comm, &handle->mype);
    SHMEM_CHECK_RET(status);
    status = MPI_Comm_size(shmemi_bootstrap_mpi_state.comm, &handle->npes);
    SHMEM_CHECK_RET(status);
    handle->allgather = shmemi_bootstrap_mpi_allgather;
    handle->alltoall = shmemi_bootstrap_mpi_alltoall;
    handle->barrier = shmemi_bootstrap_mpi_barrier;
    handle->global_exit = shmemi_bootstrap_mpi_global_exit;
    handle->finalize = shmemi_bootstrap_mpi_finalize;
    handle->pre_init_ops = NULL;
    handle->bootstrap_state = &shmemi_bootstrap_mpi_state.comm;
    return status;
}

