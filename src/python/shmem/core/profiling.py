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
from shmem.core.utils import _instance_lock

__all__ = ['get_prof', 'show_prof', 'ProfData']

ProfData = _pyshmem.ProfData


def get_prof(verbose: bool=False):
    """Return a deep-copied profiling snapshot, or ``None`` when collection is disabled for this PE."""
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a bool.")
    with _instance_lock:
        return _pyshmem.aclshmemx_get_prof(verbose)


def show_prof() -> None:
    """Print profiling data using the legacy output API.

    New code should prefer ``get_prof(verbose=True)``.
    """
    with _instance_lock:
        _pyshmem.aclshmemx_show_prof()
