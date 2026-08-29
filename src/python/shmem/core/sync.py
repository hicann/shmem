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
from shmem.core.utils import AclshmemInvalid, _instance_lock, _validate_stream

__all__ = [
    'barrier',
    'barrier_all',
    'sync',
    'sync_all',
    'barrier_on_stream',
    'barrier_all_on_stream',
    'handle_wait',
    'Handle',
    'ACLSHMEM_TEAM_WORLD',
    'ACLSHMEM_TEAM_INVALID',
    'ACLSHMEM_MAX_TEAMS',
]

Handle = _pyshmem.Handle
ACLSHMEM_TEAM_WORLD = _pyshmem.ACLSHMEM_TEAM_WORLD
ACLSHMEM_TEAM_INVALID = _pyshmem.ACLSHMEM_TEAM_INVALID
ACLSHMEM_MAX_TEAMS = _pyshmem.ACLSHMEM_MAX_TEAMS


def _validate_team(team: int) -> None:
    if (
        not isinstance(team, int)
        or isinstance(team, bool)
        or team < 0
        or team >= ACLSHMEM_MAX_TEAMS
    ):
        raise AclshmemInvalid(
            f"team must be an integer in range [0, {ACLSHMEM_MAX_TEAMS - 1}]."
        )
    if not _pyshmem._is_valid_team(team):
        raise AclshmemInvalid("team must refer to a currently active team.")


def barrier(team: int) -> None:
    """Block until all PEs in ``team`` reach the barrier."""
    with _instance_lock:
        _validate_team(team)
        _pyshmem.aclshmem_barrier(team)


def barrier_all() -> None:
    """Block until all PEs in the world team reach the barrier."""
    with _instance_lock:
        _pyshmem.aclshmem_barrier_all()


def sync(team: int) -> None:
    """Synchronize all PEs in ``team`` without adding stronger completion guarantees."""
    with _instance_lock:
        _validate_team(team)
        _pyshmem.aclshmem_sync(team)


def sync_all() -> None:
    """Synchronize all PEs in the world team."""
    with _instance_lock:
        _pyshmem.aclshmem_sync_all()


def barrier_on_stream(team: int, stream: int) -> None:
    """Enqueue a team barrier on an ACL stream without synchronizing the stream."""
    with _instance_lock:
        _validate_team(team)
        stream_addr = _validate_stream(stream, allow_none=False)
        _pyshmem.aclshmemx_barrier_on_stream(team, stream_addr)


def barrier_all_on_stream(stream: int) -> None:
    """Enqueue a world-team barrier on an ACL stream without synchronizing the stream."""
    with _instance_lock:
        stream_addr = _validate_stream(stream, allow_none=False)
        _pyshmem.aclshmemx_barrier_all_on_stream(stream_addr)


def handle_wait(handle: Handle, stream: int) -> None:
    """
    Enqueue a team-scoped completion wait and rendezvous on an ACL stream.

    Every PE in the team bound to ``handle`` must call this function in matching
    order after issuing the asynchronous operations to be covered. A world-team
    Handle therefore requires participation from every PE. The host call only
    enqueues the operation; synchronize ``stream`` before observing completion.

    Args:
        handle (Handle):
            [in] Handle bound to a currently active team.
        stream (int):
            [in] Non-bool, non-negative ACL stream address. ``0`` selects the
            default stream; ``None`` is not accepted.

    Returns:
        None: This function has no return value.

    Raises:
        AclshmemInvalid: If ``handle``, its team, or ``stream`` is invalid.
    """
    with _instance_lock:
        if not isinstance(handle, Handle):
            raise AclshmemInvalid("handle must be a Handle.")
        _validate_team(handle.team_id)
        stream_addr = _validate_stream(stream, allow_none=False)
        _pyshmem.aclshmemx_handle_wait(handle, stream_addr)
