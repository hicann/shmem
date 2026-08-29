#!/usr/bin/env python
# coding=utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
import logging
import os
from typing import Optional, Union

import shmem._pyshmem as _pyshmem
from shmem.core.utils import (
    setup_aclshmem_logger,
    AclshmemError,
    AclshmemInvalid,
    _instance_lock,
    _validate_instance_id,
)

__all__ = ['get_unique_id', 'init', 'finalize', 'get_version', 'UniqueID']

logger = logging.getLogger("aclshmem")

UniqueID = _pyshmem.UniqueId


def get_version() -> str:
    """
    Get the ACLSHMEM library version as a formatted string.

    Returns:
        str: Version string in the format ``"libaclshmem_version=X.Y"``.
    """
    major, minor = _pyshmem.aclshmem_info_get_version()
    res = f"libaclshmem_version={major}.{minor}"
    return res


def get_unique_id(empty: bool=False) -> bytes:
    """
    Create a unique ID used for UID-based ACLSHMEM initialization.

    This function generates an unique ID to initialize the aclshmem process. The ID should be generated
    by a single process (e.g., rank 0) and distributed to other processes via broadcast.

    Args:
        empty: Reserved parameter. **Ignored** in ACLSHMEM. Always treated as ``False``.

    Returns:
        bytes: The serialized native unique ID. This object can be pickled and
               sent across processes.

    Raises:
        AclshmemError: If generation of a unique ID fails.
    """
    try:
        u_id = _pyshmem.aclshmem_get_unique_id()
    except Exception as e:
        raise AclshmemError("Generate a unique ID fails.") from e

    return u_id


def init(device: int=None, uid: Optional[Union[UniqueID, bytes]]=None, rank: int=None, nranks: int=None, mpi_comm=None,
         initializer_method: str="", mem_size: int=None, instance_id: int=0) -> None:
    """
    Initialize the ACLSHMEM runtime with unique ID.

    Args:
        device: Reserved parameter. **Ignored** in ACLSHMEM. Always treated as ``None``.
        uid (aclshmem_uid, required): A unique identifier used for initialization.
        rank (int, required): The rank (0-based index) of the current process within the ACLSHMEM job.
        nranks (int, required): The total number of processes (ranks) participating in the ACLSHMEM job.
        mpi_comm: Reserved parameter. **Ignored** in ACLSHMEM. Always treated as ``None``.
        initializer_method (str): Specifies the initialization method. Must be "uid".
        mem_size (int, required): Memory size for each processing element in bytes.
        instance_id (int): ACLSHMEM instance identifier. Defaults to 0.

    Raises:
        AclshmemInvalid: If the required arguments for the selected method are missing or incorrect.
        AclshmemError: If ACLSHMEM fails to initialize using the specified method.
    """
    log_level = os.environ.get("SHMEM_LOG_LEVEL")

    if not log_level or log_level not in ["DEBUG", "INFO", "WARN", "ERROR", None]:
        logger.warning("Set log level to 'ERROR'.")
        log_level = "ERROR"
    if log_level == "WARN":
        log_level = "WARNING"
    setup_aclshmem_logger(log_level=log_level)
    if initializer_method not in ["uid"]:
        raise AclshmemInvalid("Invalid init method requested")

    if any(arg is None for arg in (uid, rank, nranks, mem_size)):
        raise AclshmemInvalid("uid, rank and nranks must be specified.")

    if not isinstance(rank, int) or isinstance(rank, bool):
        raise AclshmemInvalid("rank must be an integer.")
    if not isinstance(nranks, int) or isinstance(nranks, bool) or nranks <= 0:
        raise AclshmemInvalid("nranks must be a positive integer.")
    if rank < 0 or rank >= nranks:
        raise AclshmemInvalid("rank must satisfy 0 <= rank < nranks.")
    if not isinstance(mem_size, int) or isinstance(mem_size, bool) or mem_size <= 0:
        raise AclshmemInvalid("mem_size must be a positive integer.")
    _validate_instance_id(instance_id)

    if isinstance(uid, UniqueID):
        native_uid = uid
    elif isinstance(uid, (bytes, bytearray, memoryview)):
        try:
            native_uid = UniqueID.from_bytes(bytes(uid))
        except (TypeError, ValueError) as e:
            raise AclshmemInvalid("uid has an invalid serialized representation.") from e
    else:
        raise AclshmemInvalid("uid must be a UniqueID or serialized bytes.")

    with _instance_lock:
        attr = _pyshmem.InitAttr()
        attr.instance_id = instance_id
        ret = _pyshmem.aclshmemx_set_attr_uniqueid_args(rank, nranks, mem_size, native_uid, attr)
        if ret != 0:
            raise AclshmemError(f"ACLSHMEM set unique-ID attributes failed, ret={ret}.")

        ret = _pyshmem.aclshmemx_init_attr(_pyshmem.InitMode.UNIQUEID, attr)
        if ret != 0:
            raise AclshmemError(f"ACLSHMEM initialization fails, ret={ret}.")


def finalize(instance_id: int=None) -> None:
    """
    Finalize the ACLSHMEM runtime.

    This function should be called once per process after all ACLSHMEM operations are complete,
    typically before application exit, to release resources.

    Raises:
        AclshmemError: If the ACLSHMEM finalization fails.
    """
    if instance_id is not None:
        _validate_instance_id(instance_id)

    with _instance_lock:
        if instance_id is None:
            ret = _pyshmem.aclshmem_finalize()
        else:
            ret = _pyshmem.aclshmemx_finalize(instance_id)
        if ret != 0:
            raise AclshmemError(f"ACLSHMEM finalization fails, ret={ret}.")
