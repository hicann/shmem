# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
import torch
import torch.distributed as dist
import shmem as ash
import shmem.core as core
from shmem.core.utils import AclshmemInvalid


g_ash_size = 1024 * 1024 * 1024
g_malloc_size = 8 * 1024 * 1024
g_calloc_count = 1024
g_calloc_size = 4
g_alignment = 256
g_aligned_size = 4096

INTPTR_MAX = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8 - 1)) - 1
SIZE_T_MAX = (1 << (ctypes.sizeof(ctypes.c_size_t) * 8)) - 1


def _expect_invalid(call, case_name):
    try:
        call()
    except AclshmemInvalid:
        return
    raise AssertionError(f"[FAIL] {case_name}: expected AclshmemInvalid")


def run_memory_test():
    pe = dist.get_rank()
    world_size = dist.get_world_size()
    next = (pe + 1) % world_size
    ret = ash.set_conf_store_tls(False, "")

    # 0. disabel TLS
    if ret != 0:
        raise ValueError("[ERROR] disable tls failed.")

    # 1. get unique id
    unique_id = core.get_unique_id() if pe == 0 else None
    if pe == 0 and unique_id is None:
        raise ValueError('[ERROR] get unique id failed')
    uid_list = [unique_id]
    dist.broadcast_object_list(uid_list, src=0)
    unique_id = uid_list[0]

    # 2. init with unique id
    core.init(rank=pe, nranks=world_size, mem_size=g_ash_size, uid=unique_id, initializer_method='uid')

    # 3. malloc buffer
    aclshmem_buffer = core.buffer(g_malloc_size, mem_type=core.MemType.DEVICE_SIDE)
    print(f'pe[{pe}]: aclshmem_ptr: {aclshmem_buffer.addr} with length {type(aclshmem_buffer.length)}')
    if (
        aclshmem_buffer.addr == 0
        or aclshmem_buffer.length != g_malloc_size
        or aclshmem_buffer.mem_type != core.MemType.DEVICE_SIDE
        or not aclshmem_buffer.owned
        or aclshmem_buffer.instance_id != core.current_instance()
        or aclshmem_buffer.release_called
    ):
        raise ValueError('[ERROR] create buffer failed')

    boundary_buffer = core.Buffer(INTPTR_MAX, SIZE_T_MAX)
    if boundary_buffer.addr != INTPTR_MAX or boundary_buffer.length != SIZE_T_MAX:
        raise ValueError('[ERROR] valid Buffer integer boundaries were not preserved')

    for args, case_name in (
        ((0, 1), "zero address"),
        ((False, 1), "bool false address"),
        ((True, 1), "bool true address"),
        ((INTPTR_MAX + 1, 1), "address overflow"),
        ((1, 0), "zero length"),
        ((1, True), "bool length"),
        ((1, SIZE_T_MAX + 1), "length overflow"),
    ):
        _expect_invalid(lambda args=args: core.Buffer(*args), case_name)

    external_buffer = core.Buffer(1, 1)
    if external_buffer.owned or external_buffer.instance_id is not None or external_buffer.release_called:
        raise ValueError('[ERROR] directly constructed Buffer must be non-owning')
    try:
        core.free(external_buffer)
    except AclshmemInvalid:
        pass
    else:
        raise ValueError('[ERROR] external Buffer unexpectedly allowed free')
    if external_buffer.release_called:
        raise ValueError('[ERROR] rejected external Buffer free changed release state')

    # Exactly one thread may claim a free attempt before the native binding
    # releases the GIL.  The other thread must be rejected without entering
    # native free.
    threaded_buffer = core.buffer(4096)
    start_barrier = threading.Barrier(3)
    outcomes = []

    def free_once():
        start_barrier.wait()
        try:
            core.free(threaded_buffer)
        except AclshmemInvalid:
            outcomes.append("rejected")
        else:
            outcomes.append("called")

    threads = [threading.Thread(target=free_once) for _ in range(2)]
    for thread in threads:
        thread.start()
    start_barrier.wait()
    for thread in threads:
        thread.join()
    if sorted(outcomes) != ["called", "rejected"] or not threaded_buffer.release_called:
        raise ValueError(f'[ERROR] concurrent free guard failed: {outcomes}')

    # 4. calloc and aligned allocation with an explicit mem_type
    calloc_buffer = core.calloc(g_calloc_count, g_calloc_size, core.MemType.DEVICE_SIDE)
    if (
        calloc_buffer.addr == 0
        or calloc_buffer.length != g_calloc_count * g_calloc_size
        or calloc_buffer.mem_type != core.MemType.DEVICE_SIDE
        or not calloc_buffer.owned
        or calloc_buffer.instance_id != core.current_instance()
    ):
        raise ValueError('[ERROR] calloc buffer failed')

    aligned_buffer = core.align(g_alignment, g_aligned_size, core.MemType.DEVICE_SIDE)
    if (
        aligned_buffer.addr == 0
        or aligned_buffer.addr % g_alignment != 0
        or aligned_buffer.length != g_aligned_size
        or not aligned_buffer.owned
        or aligned_buffer.instance_id != core.current_instance()
    ):
        raise ValueError('[ERROR] aligned buffer failed')

    # HOST_SIDE heap availability depends on the installed CANN runtime.
    if os.environ.get("SHMEM_TEST_HOST_HEAP") == "1":
        host_buffer = core.buffer(4096, mem_type=core.MemType.HOST_SIDE)
        if host_buffer.addr == 0 or host_buffer.mem_type != core.MemType.HOST_SIDE:
            raise ValueError('[ERROR] host-side buffer failed')
        core.free(host_buffer)

    # 5. get next pe buffer
    next_aclshmem_buffer = core.get_peer_buffer(aclshmem_buffer, next)
    if (
        next_aclshmem_buffer.addr == 0
        or next_aclshmem_buffer.length != g_malloc_size
        or next_aclshmem_buffer.mem_type != core.MemType.DEVICE_SIDE
        or next_aclshmem_buffer.owned
        or next_aclshmem_buffer.instance_id != aclshmem_buffer.instance_id
        or next_aclshmem_buffer.release_called
    ):
        raise ValueError('[ERROR] get peer buffer failed')

    try:
        core.free(next_aclshmem_buffer)
    except AclshmemInvalid:
        pass
    else:
        raise ValueError('[ERROR] non-owning peer buffer unexpectedly allowed free')

    # 6. free buffers through the matching heap
    core.free(aligned_buffer)
    core.free(calloc_buffer)
    core.free(aclshmem_buffer)
    if not aclshmem_buffer.release_called or not next_aclshmem_buffer.release_called:
        raise ValueError('[ERROR] free state was not propagated to peer buffer')
    try:
        core.free(aclshmem_buffer)
    except AclshmemInvalid:
        pass
    else:
        raise ValueError('[ERROR] duplicate free was not rejected')

    # 7. finialize
    core.finalize()


if __name__ == "__main__":
    local_pe = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_pe)

    dist.init_process_group(backend="gloo", init_method="env://")
    run_memory_test()
    print("test_memory running success!")
