# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import ctypes
import os
import threading
from contextlib import nullcontext
from unittest import mock

import acl
import torch
import torch.distributed as dist

import shmem as ash
import shmem.core as core
from shmem.core.utils import AclshmemInvalid


G_ASH_SIZE = 1024 * 1024 * 1024
G_BUFFER_SIZE = 4096
ACL_MEMCPY_DEVICE_TO_HOST = 2


def _broadcast_unique_id(pe, root_rank=0):
    unique_id = core.get_unique_id() if dist.get_rank() == root_rank else None
    uid_list = [unique_id]
    dist.broadcast_object_list(uid_list, src=root_rank)
    return uid_list[0]


def _read_byte(addr):
    value = ctypes.c_uint8(0)
    ret = acl.rt.memcpy(
        ctypes.addressof(value),
        ctypes.sizeof(value),
        addr,
        ctypes.sizeof(value),
        ACL_MEMCPY_DEVICE_TO_HOST,
    )
    if ret != 0:
        raise RuntimeError(f"[ERROR] acl.rt.memcpy failed, ret={ret}.")
    return value.value


def _expect_invalid(call, case_name):
    try:
        call()
    except AclshmemInvalid:
        return
    raise AssertionError(f"[FAIL] {case_name}: expected AclshmemInvalid")


def _validate_prof_visibility(expected_pe, case_name):
    prof = core.get_prof(verbose=False)
    if core.direct.my_pe() == expected_pe:
        if not isinstance(prof, core.ProfData) or prof.pe_id != expected_pe:
            raise ValueError(f"[ERROR] {case_name}: expected profiling data for PE {expected_pe}.")
    elif prof is not None:
        raise ValueError(f"[ERROR] {case_name}: profiling data leaked from another instance.")


def _assert_lifecycle_waits_for_multi_instance(
    native_name, action, protected_instance, case_name
):
    entered_native = threading.Event()
    worker_done = threading.Event()
    worker_errors = []

    def native_stub(*args, **kwargs):
        entered_native.set()
        return 0

    def worker():
        try:
            action()
        except BaseException as exc:
            worker_errors.append(exc)
        finally:
            worker_done.set()

    with mock.patch.object(ash._pyshmem, native_name, side_effect=native_stub):
        if native_name == "aclshmemx_set_attr_uniqueid_args":
            init_attr_patch = mock.patch.object(
                ash._pyshmem, "aclshmemx_init_attr", return_value=0
            )
        else:
            init_attr_patch = nullcontext()

        with init_attr_patch:
            with core.multi_instance(protected_instance):
                thread = threading.Thread(target=worker)
                thread.start()
                entered_while_locked = entered_native.wait(timeout=0.5)

            thread.join(timeout=10)

    if thread.is_alive() or not worker_done.is_set():
        raise AssertionError(f"[FAIL] {case_name}: worker did not finish after context release")
    if entered_while_locked:
        raise AssertionError(
            f"[FAIL] {case_name}: lifecycle native call entered while multi_instance held the lock"
        )
    if worker_errors:
        raise worker_errors[0]
    if not entered_native.is_set():
        raise AssertionError(f"[FAIL] {case_name}: lifecycle native call was not reached")


def run_multi_instance_test():
    pe = dist.get_rank()
    world_size = dist.get_world_size()
    os.environ["SHMEM_CYCLE_PROF_PE"] = "0"
    if ash.set_conf_store_tls(False, "") != 0:
        raise ValueError("[ERROR] disable TLS failed.")

    core.init(
        rank=pe,
        nranks=world_size,
        mem_size=G_ASH_SIZE,
        uid=_broadcast_unique_id(pe),
        initializer_method="uid",
        instance_id=0,
    )
    if core.current_instance() != 0:
        raise ValueError("[ERROR] instance 0 is not active after initialization.")

    with core.multi_instance(0):
        instance_0_buffer = core.buffer(G_BUFFER_SIZE)
        if acl.rt.memset(instance_0_buffer.addr, G_BUFFER_SIZE, 0x11, G_BUFFER_SIZE) != 0:
            raise RuntimeError("[ERROR] write instance 0 heap failed.")

    instance_1_root_rank = world_size - 1
    instance_1_uid = _broadcast_unique_id(pe, root_rank=instance_1_root_rank)
    _assert_lifecycle_waits_for_multi_instance(
        "aclshmemx_set_mte_config",
        lambda: core.set_mte_config(0, 128, 0),
        0,
        "concurrent config",
    )

    _assert_lifecycle_waits_for_multi_instance(
        "aclshmemx_set_attr_uniqueid_args",
        lambda: core.init(
            rank=pe,
            nranks=world_size,
            mem_size=G_ASH_SIZE,
            uid=instance_1_uid,
            initializer_method="uid",
            instance_id=1,
        ),
        0,
        "concurrent init",
    )

    instance_1_pe = (pe + 1) % world_size
    core.init(
        rank=instance_1_pe,
        nranks=world_size,
        mem_size=G_ASH_SIZE,
        uid=instance_1_uid,
        initializer_method="uid",
        instance_id=1,
    )
    if core.current_instance() != 1:
        raise ValueError("[ERROR] instance 1 is not active after initialization.")

    with core.multi_instance(0):
        _validate_prof_visibility(0, "instance 0 profiling")
    with core.multi_instance(1):
        _validate_prof_visibility(0, "instance 1 profiling")
    with core.multi_instance(0):
        _validate_prof_visibility(0, "instance 0 profiling after restore")

    core.set_instance(0)
    if core.current_instance() != 0:
        raise ValueError("[ERROR] set_instance did not switch to instance 0.")
    core.set_instance(1)
    if core.current_instance() != 1:
        raise ValueError("[ERROR] set_instance did not switch back to instance 1.")

    with core.multi_instance(1):
        instance_1_buffer = core.buffer(G_BUFFER_SIZE)
        if acl.rt.memset(instance_1_buffer.addr, G_BUFFER_SIZE, 0x22, G_BUFFER_SIZE) != 0:
            raise RuntimeError("[ERROR] write instance 1 heap failed.")

    if instance_0_buffer.addr == instance_1_buffer.addr:
        raise ValueError("[ERROR] instance heaps unexpectedly share the same address.")
    if instance_0_buffer.instance_id != 0 or instance_1_buffer.instance_id != 1:
        raise ValueError("[ERROR] Buffer allocation instance was not recorded.")

    _expect_invalid(
        lambda: core.get_peer_buffer(instance_0_buffer, pe),
        "cross-instance peer buffer",
    )
    _expect_invalid(
        lambda: core.put(instance_0_buffer, instance_1_buffer, pe, stream=0),
        "cross-instance put",
    )
    _expect_invalid(
        lambda: core.get(instance_1_buffer, instance_0_buffer, pe, stream=0),
        "cross-instance get",
    )
    _expect_invalid(
        lambda: core.put_signal(
            instance_0_buffer,
            instance_1_buffer,
            instance_1_buffer,
            1,
            core.direct.SignalOp.SIGNAL_SET,
            pe,
        ),
        "cross-instance put_signal",
    )
    _expect_invalid(
        lambda: core.signal_op(
            instance_0_buffer,
            1,
            core.direct.SignalOp.SIGNAL_SET,
            pe,
            stream=0,
        ),
        "cross-instance signal_op",
    )
    _expect_invalid(
        lambda: core.signal_wait(
            instance_0_buffer,
            1,
            core.direct.ComparisonType.CMP_EQ,
            stream=0,
        ),
        "cross-instance signal_wait",
    )

    try:
        core.free(instance_0_buffer)
    except AclshmemInvalid:
        pass
    else:
        raise ValueError("[ERROR] cross-instance free was not rejected.")
    if instance_0_buffer.release_called:
        raise ValueError("[ERROR] rejected cross-instance free changed release state.")

    with core.multi_instance(0):
        if core.current_instance() != 0 or _read_byte(instance_0_buffer.addr) != 0x11:
            raise ValueError("[ERROR] instance 0 heap state was not preserved.")
        core.barrier_all()
        core.free(instance_0_buffer)

    if core.current_instance() != 1:
        raise ValueError("[ERROR] context manager did not restore instance 1.")

    with core.multi_instance(1):
        if _read_byte(instance_1_buffer.addr) != 0x22:
            raise ValueError("[ERROR] instance 1 heap state was not preserved.")
        core.barrier_all()
        core.free(instance_1_buffer)

    _assert_lifecycle_waits_for_multi_instance(
        "aclshmemx_finalize",
        lambda: core.finalize(instance_id=1),
        0,
        "concurrent finalize",
    )
    core.finalize(instance_id=1)
    if core.current_instance() != 0:
        raise ValueError("[ERROR] finalizing instance 1 did not fall back to instance 0.")
    core.finalize(instance_id=0)


if __name__ == "__main__":
    local_pe = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_pe)
    dist.init_process_group(backend="gloo", init_method="env://")
    run_multi_instance_test()
    print("test_multi_instance running success!")
