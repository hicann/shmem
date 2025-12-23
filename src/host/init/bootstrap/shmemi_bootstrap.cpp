/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "shmemi_host_common.h"
#include "dlfcn.h"

#define BOOTSTRAP_MODULE_MPI "aclshmem_bootstrap_mpi.so"
#define BOOTSTRAP_MODULE_UID "aclshmem_bootstrap_uid.so"

#define BOOTSTRAP_PLUGIN_INIT_FUNC "aclshmemi_bootstrap_plugin_init"
#define BOOTSTRAP_PLUGIN_PREINIT_FUNC "aclshmemi_bootstrap_plugin_pre_init"

aclshmemi_bootstrap_handle_t g_boot_handle;

static void *plugin_hdl = nullptr;
static char *plugin_name = nullptr;

int bootstrap_loader_finalize(aclshmemi_bootstrap_handle_t *handle)
{
    int status = handle->finalize(handle);

    if (status != 0)
        SHM_LOG_ERROR("Bootstrap plugin finalize failed for " << plugin_name);

    dlclose(plugin_hdl);
    plugin_hdl = nullptr;
    free(plugin_name);
    plugin_name = nullptr;

    return 0;
}



void aclshmemi_bootstrap_loader()
{
    dlerror();
    if (plugin_hdl == nullptr) {

        plugin_hdl = dlopen(plugin_name, RTLD_NOW);
    }
    const char* dl_err = dlerror();
    if (dl_err != nullptr) {
        SHM_LOG_ERROR("aclshmemi_bootstrap_loader: load bootstrap so (" << plugin_name << ") failed: " << dl_err);
        plugin_hdl = nullptr;
    }
}

void aclshmemi_bootstrap_free()
{
    if (plugin_hdl != nullptr) {
        dlclose(plugin_hdl);
        plugin_hdl = nullptr;
    }

    if (plugin_name != nullptr) {
        free(plugin_name);
        plugin_name = nullptr;
    }
}

// rank0 requires preloading uid.so to obtain the getuid capability
int32_t aclshmemi_bootstrap_pre_init(int flags, aclshmemi_bootstrap_handle_t *handle) {
    int32_t status = ACLSHMEM_SUCCESS;

    if (flags & ACLSHMEMX_INIT_WITH_MPI) {
        SHM_LOG_ERROR("Unsupport Type for bootstrap preinit.");
        return ACLSHMEM_INVALID_PARAM;
    } else if (flags & ACLSHMEMX_INIT_WITH_UNIQUEID) {
        plugin_name = BOOTSTRAP_MODULE_UID;
    } else {
        SHM_LOG_ERROR("Unknown Type for bootstrap");
        status = ACLSHMEM_INVALID_PARAM;
    }
    aclshmemi_bootstrap_loader();
   
    if (!plugin_hdl) {
        SHM_LOG_ERROR("Bootstrap unable to load " << plugin_name << ", err is: " << stderr);
        aclshmemi_bootstrap_free();
        return ACLSHMEM_INVALID_VALUE;
    }
    int (*plugin_pre_init)(aclshmemi_bootstrap_handle_t *);
    *((void **)&plugin_pre_init) = dlsym(plugin_hdl, BOOTSTRAP_PLUGIN_PREINIT_FUNC);
    if (!plugin_pre_init) {
        SHM_LOG_ERROR("Bootstrap plugin init func dlsym failed");
        aclshmemi_bootstrap_free();
        return ACLSHMEM_INNER_ERROR;
    }
    status = plugin_pre_init(&g_boot_handle);
    if (status != 0) {
        SHM_LOG_ERROR("Bootstrap plugin init failed for " << plugin_name);
        aclshmemi_bootstrap_free();
        return ACLSHMEM_INNER_ERROR;
    }
    return status;
}

void remove_tcp_prefix_and_copy(const char* input, char* output, size_t output_len) {
    memset(output, 0, output_len);
    if (output_len == 0) return;

    if (input == nullptr || strlen(input) == 0) {
        return;
    }

    const char* prefix_tcp = "tcp://";
    const char* prefix_tcp6 = "tcp6://";
    size_t len_tcp = strlen(prefix_tcp);
    size_t len_tcp6 = strlen(prefix_tcp6);
    const char* result_ptr = input;

    if (strncmp(input, prefix_tcp, len_tcp) == 0) {
        result_ptr = input + len_tcp;
    }
    else if (strncmp(input, prefix_tcp6, len_tcp6) == 0) {
        result_ptr = input + len_tcp6;
    }

    strncpy(output, result_ptr, output_len - 1);
    output[output_len - 1] = '\0';
}

int32_t aclshmemi_bootstrap_init(int flags, aclshmemx_init_attr_t *attr) {
    int32_t status = ACLSHMEM_SUCCESS;
    void *arg;
    g_boot_handle.use_attr_ipport= false;
    if (flags & ACLSHMEMX_INIT_WITH_DEFAULT){
        SHM_LOG_INFO("ACLSHMEMX_INIT_WITH_DEFAULT");
        g_boot_handle.use_attr_ipport= true;
        remove_tcp_prefix_and_copy(attr->ip_port,
            g_boot_handle.ipport,
            sizeof(g_boot_handle.ipport));
        plugin_name = BOOTSTRAP_MODULE_UID;
        arg = (attr != NULL) ? attr->comm_args : NULL;
    } else if (flags & ACLSHMEMX_INIT_WITH_MPI) {
        SHM_LOG_INFO("ACLSHMEMX_INIT_WITH_MPI");
        plugin_name = BOOTSTRAP_MODULE_MPI;
        arg = (attr != NULL) ? attr->comm_args : NULL;
    } else if (flags & ACLSHMEMX_INIT_WITH_UNIQUEID) {
        SHM_LOG_INFO("ACLSHMEMX_INIT_WITH_UNIQUEID");
        plugin_name = BOOTSTRAP_MODULE_UID;
        arg = (attr != NULL) ? attr->comm_args : NULL;
    } else {
        SHM_LOG_ERROR("Unknown Type for bootstrap");
        status = ACLSHMEM_INVALID_PARAM;
    }
    aclshmemi_bootstrap_loader();
   
    if (!plugin_hdl) {
        SHM_LOG_ERROR("Bootstrap unable to load " << plugin_name << ", err is: " << stderr);
        aclshmemi_bootstrap_free();
        return ACLSHMEM_INVALID_VALUE;
    }

    int (*plugin_init)(void *, aclshmemi_bootstrap_handle_t *);
    *((void **)&plugin_init) = dlsym(plugin_hdl, BOOTSTRAP_PLUGIN_INIT_FUNC);
    if (!plugin_init) {
        SHM_LOG_ERROR("Bootstrap plugin init func dlsym failed");
        aclshmemi_bootstrap_free();
        return ACLSHMEM_INNER_ERROR;
    }
    SHM_LOG_INFO("plugin_init");
    status = plugin_init(arg, &g_boot_handle);
    if (status != 0) {
        SHM_LOG_ERROR("Bootstrap plugin init failed for " << plugin_name);
        aclshmemi_bootstrap_free();
        return ACLSHMEM_INNER_ERROR;
    }
    g_boot_handle.is_bootstraped = true;
    return status;
}

void aclshmemi_bootstrap_finalize() {
    g_boot_handle.finalize(&g_boot_handle);
    g_boot_handle.is_bootstraped = false;
    dlclose(plugin_hdl);
}
