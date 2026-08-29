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

from contextlib import contextmanager

import shmem._pyshmem as _pyshmem
from shmem.core.utils import AclshmemError, _current_instance_id, _instance_lock, _validate_instance_id

__all__ = ['current_instance', 'set_instance', 'multi_instance', 'InstanceContext']

InstanceContext = _pyshmem.InstanceContext

def current_instance() -> int:
    """
    Return the active ACLSHMEM instance identifier.

    Returns:
        int: Identifier of the active instance.

    Raises:
        AclshmemError: If no active instance context is available.
    """
    with _instance_lock:
        return _current_instance_id()


def set_instance(instance_id: int) -> None:
    """Switch the active ACLSHMEM instance for the current process."""
    _validate_instance_id(instance_id)
    with _instance_lock:
        ret = _pyshmem.aclshmemx_instance_ctx_set(instance_id)
        if ret != 0:
            raise AclshmemError(f"Set ACLSHMEM instance context failed, ret={ret}.")


@contextmanager
def multi_instance(instance_id: int):
    """Temporarily switch instances and restore the previous instance on exit."""
    _validate_instance_id(instance_id)
    with _instance_lock:
        previous = current_instance()
        ret = _pyshmem.aclshmemx_instance_ctx_set(instance_id)
        if ret != 0:
            raise AclshmemError(f"Set ACLSHMEM instance context failed, ret={ret}.")
        try:
            yield
        finally:
            ret = _pyshmem.aclshmemx_instance_ctx_set(previous)
            if ret != 0:
                raise AclshmemError(f"Restore ACLSHMEM instance context failed, ret={ret}.")
