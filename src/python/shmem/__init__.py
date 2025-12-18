#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import os
import sys
import ctypes
import logging
import torch_npu
from ._pyshmem import (aclshmem_init, aclshmem_get_unique_id, aclshmem_init_using_unique_id, \
                       aclshmem_finialize, aclshmem_malloc, aclshmem_free, \
                       aclshmem_ptr, my_pe, pe_count, set_conf_store_tls_key, mte_set_ub_params, team_split_strided,
                       team_split_2d, team_translate_pe, \
<<<<<<< HEAD
                       team_destroy, InitAttr, OpEngineType, aclshmem_set_attributes, \
=======
                       team_destroy, InitAttr, OpEngineType, aclshmem_set_attributes, aclshmemx_set_data_op_engine_type,
                       aclshmemx_set_timeout, \
>>>>>>> 0fd3f0e... bugfix
                       InitStatus, aclshmem_calloc, aclshmem_align, aclshmemx_init_status, get_ffts_config, team_my_pe,
                       team_n_pes, \
                       aclshmem_putmem_nbi, aclshmem_getmem_nbi, aclshmem_putmem, aclshmem_getmem, aclshmemx_putmem_signal,
                       aclshmemx_putmem_signal_nbi, \
                       aclshmem_info_get_version, aclshmem_info_get_name, \
                       aclshmem_team_get_config, OptionalAttr, aclshmem_global_exit, set_conf_store_tls, set_log_level,
                       set_extern_logger)

current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
sys.path.append(current_dir)
libs_path = os.path.join(os.getenv('ACLSHMEM_HOME_PATH', current_dir), 'shmem/lib')
for lib in ["libshmem.so"]:
    lib_path = os.path.join(libs_path, lib)
    try:
        ctypes.CDLL(lib_path)
    except OSError as e:
        logging.error(f"Failed to load shared library: {lib_path}, {e}")

__all__ = [
    'aclshmem_init',
    'aclshmem_get_unique_id',
    'aclshmem_init_using_unique_id',
    'aclshmem_finialize',
    'aclshmem_malloc',
    'aclshmem_free',
    'aclshmem_ptr',
    'my_pe',
    'pe_count',
    'set_conf_store_tls_key',
    'set_conf_store_tls',
    'team_my_pe',
    'team_n_pes',
    'mte_set_ub_params',
    'team_split_strided',
    'team_split_2d',
    'team_translate_pe',
    'team_destroy',
    'InitAttr',
    'InitStatus',
    'OpEngineType',
    'aclshmem_set_attributes',
<<<<<<< HEAD
=======
    'aclshmemx_set_data_op_engine_type',
    'aclshmemx_set_timeout',
>>>>>>> 0fd3f0e... bugfix
    'aclshmem_calloc',
    'aclshmem_align',
    'aclshmemx_init_status',
    'get_ffts_config',
    'aclshmem_global_exit',
    'aclshmem_putmem_nbi',
    'aclshmem_getmem_nbi',
    'aclshmemx_putmem_signal',
    'aclshmemx_putmem_signal_nbi',
    'aclshmem_putmem',
    'aclshmem_getmem',
    'aclshmem_info_get_version',
    'aclshmem_info_get_name',
    'aclshmem_team_get_config',
    'set_log_level',
    'set_extern_logger'
]