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
import shmem._pyshmem as _pyshmem
from shmem.core.utils import (
    Buffer,
    AclshmemInvalid,
    _instance_lock,
    _validate_buffer_instance,
    _validate_stream,
)
from shmem.core.direct import ComparisonType, SignalOp

__all__ = ['put_signal', 'signal_op', 'signal_wait', 'put', 'get', 'quiet']

_SIGNAL_WORD_SIZE = 4
_SUPPORTED_SIGNAL_OPERATIONS = (SignalOp.SIGNAL_SET, SignalOp.SIGNAL_ADD)
_SUPPORTED_COMPARISON_OPERATIONS = (
    ComparisonType.CMP_EQ,
    ComparisonType.CMP_NE,
    ComparisonType.CMP_GT,
    ComparisonType.CMP_GE,
    ComparisonType.CMP_LT,
    ComparisonType.CMP_LE,
)


def _validate_buffer(buf: Buffer, name: str) -> None:
    if not isinstance(buf, Buffer):
        raise AclshmemInvalid(f"{name} must be a Buffer.")
    buf._ensure_usable()
    _validate_buffer_instance(buf, name)


def _validate_transfer_capacity(dst: Buffer, src: Buffer) -> None:
    if dst.length < src.length:
        raise AclshmemInvalid(
            f"dst length ({dst.length}) must be greater than or equal to "
            f"src length ({src.length})."
        )


def _validate_signal_buffer(signal_var: Buffer) -> None:
    _validate_buffer(signal_var, "signal_var")
    if signal_var.length < _SIGNAL_WORD_SIZE:
        raise AclshmemInvalid(
            f"signal_var must contain at least {_SIGNAL_WORD_SIZE} bytes."
        )
    if signal_var.addr % _SIGNAL_WORD_SIZE != 0:
        raise AclshmemInvalid(
            f"signal_var address must be {_SIGNAL_WORD_SIZE}-byte aligned."
        )


def _validate_remote_pe(remote_pe: int) -> None:
    if not isinstance(remote_pe, int) or isinstance(remote_pe, bool):
        raise AclshmemInvalid("remote_pe must be an integer.")
    pe_count = _pyshmem.pe_count()
    if pe_count <= 0:
        raise AclshmemInvalid("ACLSHMEM must be initialized before validating remote_pe.")
    if remote_pe < 0 or remote_pe >= pe_count:
        raise AclshmemInvalid(f"remote_pe must satisfy 0 <= remote_pe < {pe_count}.")


def _validate_signal_operation(signal_operation) -> None:
    if (
        not isinstance(signal_operation, SignalOp)
        or signal_operation not in _SUPPORTED_SIGNAL_OPERATIONS
    ):
        raise AclshmemInvalid(
            "signal_operation must be SignalOp.SIGNAL_SET or SignalOp.SIGNAL_ADD."
        )


def _validate_comparison_operation(signal_operation) -> None:
    if (
        not isinstance(signal_operation, ComparisonType)
        or signal_operation not in _SUPPORTED_COMPARISON_OPERATIONS
    ):
        raise AclshmemInvalid(
            "signal_operation must be a supported ComparisonType value."
        )


def put_signal(dst: Buffer, src: Buffer, signal_var: Buffer, signal_val: int, signal_operation: SignalOp,
               remote_pe: int, stream=None) -> None:
    """
    Synchronous (blocking) interface. Copy contiguous data from the local PE to a
    symmetric memory address on the specified PE, and update a remote signal variable
    on completion.

    Args:
        dst (Buffer):
            [in] Symmetric address of the destination data on the remote PE.
        src (Buffer):
            [in] Local memory of the source data.
        signal_var (Buffer):
            [in] Symmetric address of the signal word to be updated on the remote PE.
        signal_val (int):
            [in] The value used to update the signal variable.
        signal_operation (SignalOp):
            [in] Must be a ``SignalOp`` value used to update the signal variable.
            Supported: ``SignalOp.SIGNAL_SET`` / ``SignalOp.SIGNAL_ADD``.
        remote_pe (int):
            [in] PE number of the remote PE. Must satisfy
            ``0 <= remote_pe < pe_count()``.
        stream:
            [in] Reserved parameter, ignored. This interface does not provide
            explicit stream ordering.

    Returns:
        None: This function has no return value.

    Raises:
        AclshmemInvalid: If a buffer is unusable, the destination is too small,
            the signal word is too short or misaligned, the signal operation has
            the wrong type, or ``remote_pe`` is invalid.
    """
    with _instance_lock:
        _validate_buffer(dst, "dst")
        _validate_buffer(src, "src")
        _validate_transfer_capacity(dst, src)
        _validate_signal_buffer(signal_var)
        _validate_remote_pe(remote_pe)
        _validate_signal_operation(signal_operation)
        _pyshmem.aclshmemx_putmem_signal(
            dst.addr, src.addr, src.length, signal_var.addr, signal_val, signal_operation, remote_pe
        )


def signal_op(signal_var: Buffer, signal_val: int, signal_operation: SignalOp, remote_pe: int,
              stream: int=None) -> None:
    """
    Non-blocking interface. Performs an atomic operation on a remote signal variable
    at the specified PE, with the operation executed on the given stream. The caller
    must synchronize the stream to observe the result.

    Args:
        signal_var (Buffer):
            [in] Local address of the signal variable that is accessible at the target PE.
        signal_val (int):
            [in] The value to be used in the atomic operation.
        signal_operation (SignalOp):
            [in] Must be a ``SignalOp`` value for the remote signal.
            Supported: ``SignalOp.SIGNAL_SET`` / ``SignalOp.SIGNAL_ADD``.
        remote_pe (int):
            [in] The PE number on which the remote signal variable is to be updated.
            Must satisfy ``0 <= remote_pe < pe_count()``.
        stream (int):
            [in] Non-bool, non-negative ACL stream address. ``0`` is the default
            stream; ``None`` is not accepted.

    Returns:
        None: This function has no return value.

    Raises:
        AclshmemInvalid: If the signal word, signal operation, ``remote_pe``,
            or stream is invalid.
    """
    with _instance_lock:
        _validate_signal_buffer(signal_var)
        _validate_remote_pe(remote_pe)
        _validate_signal_operation(signal_operation)
        stream_addr = _validate_stream(stream, allow_none=False)
        _pyshmem.aclshmemx_signal_op_on_stream(
            signal_var.addr, signal_val, signal_operation, remote_pe, stream_addr
        )


def signal_wait(signal_var: Buffer, signal_val: int, signal_operation: ComparisonType, stream: int) -> None:
    """
    Waits until a symmetric signal variable satisfies a given condition. The wait
    is performed on the specified stream; the call returns immediately on the host.
    When the stream is synchronized, the condition
    ``signal_var`` ``cmp`` ``signal_val`` is guaranteed to be true.

    .. note::
        This operation waits on an address local to the calling device; it does
        not provide a cross-machine remote-wait operation.

    Args:
        signal_var (Buffer):
            [in] Local address of the source signal variable.
        signal_val (int):
            [in] The value against which the object pointed to by signal_var will be
            compared.
        signal_operation (ComparisonType):
            [in] Must be a ``ComparisonType`` value used to evaluate the condition.
            Supported: ``ComparisonType.CMP_EQ`` / ``CMP_NE`` / ``CMP_GT`` /
            ``CMP_GE`` / ``CMP_LT`` / ``CMP_LE``.
        stream (int):
            [in] Non-bool, non-negative ACL stream address. ``0`` is the default
            stream; ``None`` is not accepted.

    Returns:
        None: This function has no return value.

    Raises:
        AclshmemInvalid: If the signal word, comparison operation, or stream is invalid.
    """
    with _instance_lock:
        _validate_signal_buffer(signal_var)
        _validate_comparison_operation(signal_operation)
        stream_addr = _validate_stream(stream, allow_none=False)
        _pyshmem.aclshmemx_signal_wait_until_on_stream(
            signal_var.addr, signal_operation, signal_val, stream_addr
        )


def put(dst: Buffer, src: Buffer, remote_pe: int, stream: int=None) -> None:
    """
    Non-blocking interface. Copy contiguous data from the local PE to a symmetric
    memory address on a remote PE, ordered on the given stream. The caller must
    synchronize the stream to ensure the transfer is complete.

    Args:
        dst (Buffer):
            [in] Symmetric address of the destination data on the remote PE.
        src (Buffer):
            [in] Local memory of the source data.
        remote_pe (int):
            [in] PE number of the remote PE. Must satisfy
            ``0 <= remote_pe < pe_count()``.
        stream (int):
            [in] Non-bool, non-negative ACL stream address. Passing ``0`` or
            ``None`` uses the default stream.

    Returns:
        None: This function has no return value.

    Raises:
        AclshmemInvalid: If a buffer is unusable, the destination is too small,
            ``remote_pe`` is invalid, or stream is invalid.
    """
    with _instance_lock:
        _validate_buffer(dst, "dst")
        _validate_buffer(src, "src")
        _validate_transfer_capacity(dst, src)
        _validate_remote_pe(remote_pe)
        stream_addr = _validate_stream(stream, allow_none=True)
        _pyshmem.aclshmemx_putmem_on_stream(dst.addr, src.addr, src.length, remote_pe, stream_addr)


def get(dst: Buffer, src: Buffer, remote_pe: int, stream: int=None) -> None:
    """
    Non-blocking interface. Copy contiguous data from symmetric memory on a remote PE
    to a local buffer, ordered on the given stream. The caller must synchronize the
    stream to ensure the transfer is complete.

    Args:
        dst (Buffer):
            [in] Local memory of the destination data.
        src (Buffer):
            [in] Symmetric address of the source data on the remote PE.
        remote_pe (int):
            [in] PE number of the remote PE. Must satisfy
            ``0 <= remote_pe < pe_count()``.
        stream (int):
            [in] Non-bool, non-negative ACL stream address. Passing ``0`` or
            ``None`` uses the default stream.

    Returns:
        None: This function has no return value.

    Raises:
        AclshmemInvalid: If a buffer is unusable, the destination is too small,
            ``remote_pe`` is invalid, or stream is invalid.
    """
    with _instance_lock:
        _validate_buffer(dst, "dst")
        _validate_buffer(src, "src")
        _validate_transfer_capacity(dst, src)
        _validate_remote_pe(remote_pe)
        stream_addr = _validate_stream(stream, allow_none=True)
        _pyshmem.aclshmemx_getmem_on_stream(dst.addr, src.addr, src.length, remote_pe, stream_addr)


def quiet(stream: int) -> None:
    """
    Ensures completion of all previously issued operations on symmetric data
    objects on the given stream. The quiet is queued on the specified stream;
    the caller must synchronize the stream to observe completion from the host.

    Args:
        stream (int):
            [in] Non-bool, non-negative ACL stream address. ``0`` is the default
            stream; ``None`` is not accepted.

    Returns:
        None: This function has no return value.

    Raises:
        AclshmemInvalid: If stream is invalid.
    """
    with _instance_lock:
        stream_addr = _validate_stream(stream, allow_none=False)
        _pyshmem.aclshmemx_quiet_on_stream(stream_addr)
