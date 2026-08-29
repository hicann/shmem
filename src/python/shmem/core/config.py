#!/usr/bin/env python
# coding=utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import shmem._pyshmem as _pyshmem
from shmem.core.utils import AclshmemError, AclshmemInvalid, _instance_lock

__all__ = ['set_mte_config', 'set_sdma_config', 'set_rdma_config', 'set_udma_config']

_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1


def _validate_config(offset: int, ub_size: int, sync_id: int, min_ub_size: int=0) -> None:
    values = (
        ("offset", offset, 0, _UINT64_MAX),
        ("ub_size", ub_size, min_ub_size, _UINT32_MAX),
        ("sync_id", sync_id, 0, _UINT32_MAX),
    )
    for name, value, minimum, maximum in values:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < minimum
            or value > maximum
        ):
            raise AclshmemInvalid(
                f"{name} must be an integer in [{minimum}, {maximum}]."
            )


def _set_config(
    name: str,
    function,
    offset: int,
    ub_size: int,
    sync_id: int,
    min_ub_size: int=0,
) -> None:
    with _instance_lock:
        _validate_config(offset, ub_size, sync_id, min_ub_size)
        ret = function(offset, ub_size, sync_id)
        if ret != 0:
            raise AclshmemError(f"Set {name} configuration failed, ret={ret}.")


def set_mte_config(offset: int, ub_size: int, sync_id: int) -> None:
    """
    Configure MTE workspace parameters for the active SHMEM instance.

    Call this function after initializing and selecting the target instance and
    before issuing operations that use the configuration.

    Args:
        offset (int):
            [in] Workspace offset/address in bytes. Must be a non-bool integer
            in ``[0, 2**64 - 1]``.
        ub_size (int):
            [in] Workspace size in bytes. Must be a non-bool integer in
            ``[0, 2**32 - 1]``.
        sync_id (int):
            [in] Synchronization identifier. Must be a non-bool integer in
            ``[0, 2**32 - 1]``.

    Returns:
        None: This function has no return value.

    Raises:
        AclshmemInvalid: If an argument is outside its accepted range.
        AclshmemError: If the active instance cannot apply the configuration.
    """
    _set_config("MTE", _pyshmem.aclshmemx_set_mte_config, offset, ub_size, sync_id)


def set_sdma_config(offset: int, ub_size: int, sync_id: int) -> None:
    """
    Configure SDMA workspace parameters for the active SHMEM instance.

    Call this function after initializing and selecting the target instance and
    before issuing operations that use the configuration. Availability depends
    on the current platform and runtime.

    Args:
        offset (int):
            [in] Workspace offset/address in bytes. Must be a non-bool integer
            in ``[0, 2**64 - 1]``.
        ub_size (int):
            [in] Workspace size in bytes. Must be a non-bool integer in
            ``[0, 2**32 - 1]``.
        sync_id (int):
            [in] Synchronization identifier. Must be a non-bool integer in
            ``[0, 2**32 - 1]``.

    Returns:
        None: This function has no return value.

    Raises:
        AclshmemInvalid: If an argument is outside its accepted range.
        AclshmemError: If the active instance cannot apply the configuration.
    """
    _set_config("SDMA", _pyshmem.aclshmemx_set_sdma_config, offset, ub_size, sync_id)


def set_rdma_config(offset: int, ub_size: int, sync_id: int) -> None:
    """
    Configure RDMA workspace parameters for the active SHMEM instance.

    Call this function after initializing and selecting the target instance and
    before issuing operations that use the configuration. Availability depends
    on the current platform and runtime.

    Args:
        offset (int):
            [in] Workspace offset/address in bytes. Must be a non-bool integer
            in ``[0, 2**64 - 1]``.
        ub_size (int):
            [in] Workspace size in bytes. Must be a non-bool integer in
            ``[128, 2**32 - 1]``.
        sync_id (int):
            [in] Synchronization identifier. Must be a non-bool integer in
            ``[0, 2**32 - 1]``.

    Returns:
        None: This function has no return value.

    Raises:
        AclshmemInvalid: If an argument is outside its accepted range.
        AclshmemError: If the active instance cannot apply the configuration.
    """
    _set_config(
        "RDMA",
        _pyshmem.aclshmemx_set_rdma_config,
        offset,
        ub_size,
        sync_id,
        min_ub_size=128,
    )


def set_udma_config(offset: int, ub_size: int, sync_id: int) -> None:
    """
    Configure UDMA workspace parameters for the active SHMEM instance.

    Call this function after initializing and selecting the target instance and
    before issuing operations that use the configuration. Availability depends
    on the current platform and runtime.

    Args:
        offset (int):
            [in] Workspace offset/address in bytes. Must be a non-bool integer
            in ``[0, 2**64 - 1]``.
        ub_size (int):
            [in] Workspace size in bytes. Must be a non-bool integer in
            ``[128, 2**32 - 1]``.
        sync_id (int):
            [in] Synchronization identifier. Must be a non-bool integer in
            ``[0, 2**32 - 1]``.

    Returns:
        None: This function has no return value.

    Raises:
        AclshmemInvalid: If an argument is outside its accepted range.
        AclshmemError: If the active instance cannot apply the configuration.
    """
    _set_config(
        "UDMA",
        _pyshmem.aclshmemx_set_udma_config,
        offset,
        ub_size,
        sync_id,
        min_ub_size=128,
    )
