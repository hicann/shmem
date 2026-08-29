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

import acl
import torch
import torch.distributed as dist

import shmem as ash
import shmem.core as core


G_ASH_SIZE = 1024 * 1024 * 1024
G_BUFFER_SIZE = 4096
ACL_MEMCPY_DEVICE_TO_HOST = 2


def _broadcast_unique_id(pe):
    unique_id = core.get_unique_id() if pe == 0 else None
    uid_list = [unique_id]
    dist.broadcast_object_list(uid_list, src=0)
    return uid_list[0]


def _init(pe, world_size, unique_id):
    engine = os.environ.get("SHMEM_TEST_ENGINE", "MTE").upper()
    if engine == "MTE":
        core.init(
            rank=pe,
            nranks=world_size,
            mem_size=G_ASH_SIZE,
            uid=unique_id,
            initializer_method="uid",
        )
        return
    if engine != "ROCE":
        raise ValueError("SHMEM_TEST_ENGINE must be MTE or ROCE.")

    attr = ash.InitAttr()
    native_uid = ash.UniqueId.from_bytes(unique_id)
    ret = ash.aclshmemx_set_attr_uniqueid_args(pe, world_size, G_ASH_SIZE, native_uid, attr)
    if ret != 0:
        raise RuntimeError(f"[ERROR] construct ROCE InitAttr failed, ret={ret}.")
    option_attr = attr.option_attr
    option_attr.data_op_engine_type = ash.OpEngineType.ROCE
    attr.option_attr = option_attr
    ret = ash.aclshmemx_init_attr(ash.InitMode.UNIQUEID, attr)
    if ret != 0:
        raise RuntimeError(f"[ERROR] ROCE initialization failed, ret={ret}.")


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


def run_handle_wait_test():
    pe = dist.get_rank()
    world_size = dist.get_world_size()
    next_pe = (pe + 1) % world_size
    previous_pe = (pe - 1 + world_size) % world_size
    if ash.set_conf_store_tls(False, "") != 0:
        raise ValueError("[ERROR] disable TLS failed.")

    _init(pe, world_size, _broadcast_unique_id(pe))
    send_buffer = core.buffer(G_BUFFER_SIZE)
    recv_buffer = core.buffer(G_BUFFER_SIZE)
    if acl.rt.memset(send_buffer.addr, G_BUFFER_SIZE, pe + 1, G_BUFFER_SIZE) != 0:
        raise RuntimeError("[ERROR] initialize send buffer failed.")
    if acl.rt.memset(recv_buffer.addr, G_BUFFER_SIZE, 0, G_BUFFER_SIZE) != 0:
        raise RuntimeError("[ERROR] initialize receive buffer failed.")

    stream = int(torch.npu.current_stream().npu_stream)
    core.put(recv_buffer, send_buffer, remote_pe=next_pe, stream=stream)
    core.handle_wait(core.Handle(core.ACLSHMEM_TEAM_WORLD), stream)
    torch.npu.synchronize()
    dist.barrier()

    expected = previous_pe + 1
    actual = _read_byte(recv_buffer.addr)
    if actual != expected:
        raise ValueError(f"[ERROR] handle_wait visibility check failed: expected {expected}, got {actual}.")

    core.free(recv_buffer)
    core.free(send_buffer)
    core.finalize()


if __name__ == "__main__":
    local_pe = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_pe)
    dist.init_process_group(backend="gloo", init_method="env://")
    run_handle_wait_test()
    print("test_handle_wait running success!")
