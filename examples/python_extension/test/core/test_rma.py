# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
import ctypes
import torch
import torch.distributed as dist
import acl
import shmem as ash
import shmem.core as core
from shmem.core.utils import AclshmemInvalid


g_ash_size = 1024 * 1024 * 1024
g_malloc_size = 8 * 1024 * 1024
g_signal_size = 4
g_value = 1
g_sig_value = 2

# ACL memory copy direction constants (acl Python module does not export named enums)
ACL_MEMCPY_HOST_TO_HOST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3


def _read_int32_from_device(addr):
    """Read a single int32 value from device memory at the given address."""
    host_val = ctypes.c_int32(0)
    ret = acl.rt.memcpy(ctypes.addressof(host_val), 4, addr, 4,
                        ACL_MEMCPY_DEVICE_TO_HOST)
    if ret != 0:
        raise RuntimeError(f"[ERROR] acl.rt.memcpy failed, ret={ret}")
    return host_val.value


def _expect_invalid(call, case_name):
    try:
        call()
    except AclshmemInvalid:
        return
    raise AssertionError(f"[FAIL] {case_name}: expected AclshmemInvalid")


def check_acl(ret, operation):
    if ret != 0:
        raise RuntimeError(f"[ERROR] {operation} failed, ret={ret}")


def run_put_signal_test():
    pe = dist.get_rank()
    world_size = dist.get_world_size()
    next_pe = (pe + 1) % world_size
    prev_pe = (pe - 1 + world_size) % world_size
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
    send_aclshmem_buffer = core.buffer(g_malloc_size)
    if (send_aclshmem_buffer.addr is None) or (send_aclshmem_buffer.length != g_malloc_size):
        raise ValueError('[ERROR] create send buffer failed')
    check_acl(
        acl.rt.memset(send_aclshmem_buffer.addr, g_malloc_size, 0, g_malloc_size),
        "initialize send buffer",
    )

    recv_aclshmem_buffer = core.buffer(g_malloc_size)
    if (recv_aclshmem_buffer.addr is None) or (recv_aclshmem_buffer.length != g_malloc_size):
        raise ValueError('[ERROR] create recv buffer failed')
    check_acl(
        acl.rt.memset(recv_aclshmem_buffer.addr, g_malloc_size, 0, g_malloc_size),
        "initialize receive buffer",
    )

    signal_aclshmem_buffer = core.buffer(g_signal_size)
    if (signal_aclshmem_buffer.addr is None) or (signal_aclshmem_buffer.length != g_signal_size):
        raise ValueError('[ERROR] create signal buffer failed')
    check_acl(
        acl.rt.memset(signal_aclshmem_buffer.addr, g_signal_size, 0, g_signal_size),
        "initialize signal buffer",
    )

    # 4. Invalid RMA arguments must be rejected before entering native code.
    small_dst = core.Buffer._borrowed(
        recv_aclshmem_buffer.addr,
        send_aclshmem_buffer.length - 1,
        recv_aclshmem_buffer.mem_type,
        recv_aclshmem_buffer,
    )
    short_signal = core.Buffer._borrowed(
        signal_aclshmem_buffer.addr,
        g_signal_size - 1,
        signal_aclshmem_buffer.mem_type,
        signal_aclshmem_buffer,
    )
    misaligned_signal = core.Buffer._borrowed(
        recv_aclshmem_buffer.addr + 1,
        g_signal_size,
        recv_aclshmem_buffer.mem_type,
        recv_aclshmem_buffer,
    )

    _expect_invalid(
        lambda: core.put_signal(
            small_dst,
            send_aclshmem_buffer,
            signal_aclshmem_buffer,
            g_sig_value,
            core.direct.SignalOp.SIGNAL_SET,
            next_pe,
        ),
        "put_signal destination capacity",
    )
    _expect_invalid(
        lambda: core.put(small_dst, send_aclshmem_buffer, next_pe, stream=0),
        "put destination capacity",
    )
    _expect_invalid(
        lambda: core.get(small_dst, send_aclshmem_buffer, next_pe, stream=0),
        "get destination capacity",
    )
    _expect_invalid(
        lambda: core.put_signal(
            recv_aclshmem_buffer,
            send_aclshmem_buffer,
            short_signal,
            g_sig_value,
            core.direct.SignalOp.SIGNAL_SET,
            next_pe,
        ),
        "put_signal signal word length",
    )
    _expect_invalid(
        lambda: core.signal_op(
            misaligned_signal,
            g_sig_value,
            core.direct.SignalOp.SIGNAL_SET,
            pe,
            stream=0,
        ),
        "signal_op signal word alignment",
    )
    _expect_invalid(
        lambda: core.signal_wait(
            short_signal,
            g_sig_value,
            core.direct.ComparisonType.CMP_EQ,
            stream=0,
        ),
        "signal_wait signal word length",
    )
    _expect_invalid(
        lambda: core.put_signal(
            recv_aclshmem_buffer,
            send_aclshmem_buffer,
            signal_aclshmem_buffer,
            g_sig_value,
            core.direct.SignalOp.SIGNAL_SET,
            -1,
        ),
        "put_signal negative remote_pe",
    )
    _expect_invalid(
        lambda: core.put(recv_aclshmem_buffer, send_aclshmem_buffer, True, stream=0),
        "put bool remote_pe",
    )
    _expect_invalid(
        lambda: core.get(recv_aclshmem_buffer, send_aclshmem_buffer, world_size, stream=0),
        "get out-of-range remote_pe",
    )
    _expect_invalid(
        lambda: core.signal_op(
            signal_aclshmem_buffer,
            g_sig_value,
            core.direct.SignalOp.SIGNAL_SET,
            -1,
            stream=0,
        ),
        "signal_op negative remote_pe",
    )
    _expect_invalid(
        lambda: core.put(recv_aclshmem_buffer, send_aclshmem_buffer, next_pe, stream=True),
        "put bool stream",
    )
    _expect_invalid(
        lambda: core.get(recv_aclshmem_buffer, send_aclshmem_buffer, next_pe, stream=-1),
        "get negative stream",
    )
    _expect_invalid(
        lambda: core.signal_op(
            signal_aclshmem_buffer,
            g_sig_value,
            core.direct.SignalOp.SIGNAL_SET,
            pe,
            stream="invalid",
        ),
        "signal_op invalid stream handle",
    )
    _expect_invalid(
        lambda: core.signal_wait(
            signal_aclshmem_buffer,
            g_sig_value,
            core.direct.ComparisonType.CMP_EQ,
            stream=True,
        ),
        "signal_wait bool stream",
    )
    _expect_invalid(lambda: core.quiet(stream=-1), "quiet negative stream")
    _expect_invalid(
        lambda: core.put_signal(
            recv_aclshmem_buffer,
            send_aclshmem_buffer,
            signal_aclshmem_buffer,
            g_sig_value,
            999,
            next_pe,
        ),
        "put_signal integer signal operation",
    )
    _expect_invalid(
        lambda: core.signal_op(
            signal_aclshmem_buffer,
            g_sig_value,
            core.direct.ComparisonType.CMP_EQ,
            pe,
            stream=0,
        ),
        "signal_op wrong enum type",
    )
    _expect_invalid(
        lambda: core.signal_wait(
            signal_aclshmem_buffer,
            g_sig_value,
            999,
            stream=0,
        ),
        "signal_wait integer comparison operation",
    )
    _expect_invalid(
        lambda: core.signal_wait(
            signal_aclshmem_buffer,
            g_sig_value,
            core.direct.SignalOp.SIGNAL_SET,
            stream=0,
        ),
        "signal_wait wrong enum type",
    )

    # 5. write known value to local send buffer (scalar put to own PE)
    ash._pyshmem.aclshmem_int32_p(send_aclshmem_buffer.addr, g_value * pe, pe)
    core.quiet(stream=0)
    torch.npu.synchronize()
    send_val = _read_int32_from_device(send_aclshmem_buffer.addr)
    assert send_val == g_value * pe, \
        f"[FAIL] step 5: expected send buffer {g_value * pe}, got {send_val}"
    # All target buffers must be initialized before any PE starts a remote
    # write.  Without this barrier a slower PE can clear its receive buffer
    # after a faster peer has already written to it.
    dist.barrier()

    # 6. put_signal: copy local send buffer to next PE's recv buffer, set signal on next PE
    core.put_signal(recv_aclshmem_buffer, send_aclshmem_buffer, signal_aclshmem_buffer, g_sig_value,
                    core.direct.SignalOp.SIGNAL_SET, next_pe)
    core.quiet(stream=0)
    torch.npu.synchronize()
    dist.barrier()
    signal_val = _read_int32_from_device(signal_aclshmem_buffer.addr)
    assert signal_val == g_sig_value, \
        f"[FAIL] step 6: expected signal {g_sig_value}, got {signal_val} on {pe=}"
    recv_val = _read_int32_from_device(recv_aclshmem_buffer.addr)
    assert recv_val == g_value * prev_pe, \
        f"[FAIL] step 6: expected recv buffer {g_value * prev_pe} from PE {prev_pe}, got {recv_val}"

    # 7. signal_op + signal_wait: atomic signal op on own PE, then wait for it
    stream, ret = acl.rt.create_stream()
    check_acl(ret, "step 7 acl.rt.create_stream")
    check_acl(
        acl.rt.memset(signal_aclshmem_buffer.addr, g_signal_size, 0, g_signal_size),
        "step 7 initialize signal buffer",
    )
    core.signal_op(signal_aclshmem_buffer, g_sig_value, core.direct.SignalOp.SIGNAL_SET, pe, stream=stream)
    core.signal_wait(signal_aclshmem_buffer, g_sig_value, core.direct.ComparisonType.CMP_EQ, stream=stream)
    check_acl(acl.rt.synchronize_stream(stream), "step 7 acl.rt.synchronize_stream")
    signal_val = _read_int32_from_device(signal_aclshmem_buffer.addr)
    assert signal_val == g_sig_value, \
        f"[FAIL] step 7: expected signal {g_sig_value} after signal_op, got {signal_val}"
    check_acl(acl.rt.destroy_stream(stream), "step 7 acl.rt.destroy_stream")

    # 8. put: non-blocking put to next PE on a stream
    stream, ret = acl.rt.create_stream()
    check_acl(ret, "step 8 acl.rt.create_stream")
    check_acl(
        acl.rt.memset(send_aclshmem_buffer.addr, g_malloc_size, 0, g_malloc_size),
        "step 8 initialize send buffer",
    )
    check_acl(
        acl.rt.memset(recv_aclshmem_buffer.addr, g_malloc_size, 0, g_malloc_size),
        "step 8 initialize receive buffer",
    )
    ash._pyshmem.aclshmem_int32_p(send_aclshmem_buffer.addr, g_value * pe, pe)
    core.quiet(stream=0)
    torch.npu.synchronize()
    dist.barrier()
    core.put(recv_aclshmem_buffer, send_aclshmem_buffer, next_pe, stream)
    check_acl(acl.rt.synchronize_stream(stream), "step 8 acl.rt.synchronize_stream")
    check_acl(acl.rt.destroy_stream(stream), "step 8 acl.rt.destroy_stream")
    dist.barrier()
    recv_val = _read_int32_from_device(recv_aclshmem_buffer.addr)
    assert recv_val == g_value * prev_pe, \
        f"[FAIL] step 8: expected recv buffer {g_value * prev_pe} from PE {prev_pe}, got {recv_val}"

    # 9. get: non-blocking get from next PE on a stream
    stream, ret = acl.rt.create_stream()
    check_acl(ret, "step 9 acl.rt.create_stream")
    check_acl(
        acl.rt.memset(send_aclshmem_buffer.addr, g_malloc_size, 0, g_malloc_size),
        "step 9 initialize send buffer",
    )
    check_acl(
        acl.rt.memset(recv_aclshmem_buffer.addr, g_malloc_size, 0, g_malloc_size),
        "step 9 initialize receive buffer",
    )
    dist.barrier()
    ash._pyshmem.aclshmem_int32_p(send_aclshmem_buffer.addr, g_value * next_pe, next_pe)
    core.quiet(stream=0)
    torch.npu.synchronize()
    dist.barrier()
    core.get(recv_aclshmem_buffer, send_aclshmem_buffer, next_pe, stream)
    check_acl(acl.rt.synchronize_stream(stream), "step 9 acl.rt.synchronize_stream")
    check_acl(acl.rt.destroy_stream(stream), "step 9 acl.rt.destroy_stream")
    recv_val = _read_int32_from_device(recv_aclshmem_buffer.addr)
    assert recv_val == g_value * next_pe, \
        f"[FAIL] step 9: expected recv buffer {g_value * next_pe} from PE {next_pe}, got {recv_val}"

    # 10. quiet: ensure all outstanding RMA operations complete before verifying
    check_acl(
        acl.rt.memset(send_aclshmem_buffer.addr, g_malloc_size, 0, g_malloc_size),
        "step 10 initialize send buffer",
    )
    check_acl(
        acl.rt.memset(recv_aclshmem_buffer.addr, g_malloc_size, 0, g_malloc_size),
        "step 10 initialize receive buffer",
    )
    ash._pyshmem.aclshmem_int32_p(send_aclshmem_buffer.addr, g_value * pe, pe)
    core.quiet(stream=0)
    torch.npu.synchronize()
    dist.barrier()
    core.put(recv_aclshmem_buffer, send_aclshmem_buffer, next_pe, stream=0)
    core.quiet(stream=0)
    torch.npu.synchronize()
    dist.barrier()
    recv_val = _read_int32_from_device(recv_aclshmem_buffer.addr)
    assert recv_val == g_value * prev_pe, \
        f"[FAIL] step 10: expected recv buffer {g_value * prev_pe} from PE {prev_pe}, got {recv_val}"

    # 11. quiet_on_stream: explicit stream put + quiet_on_stream + synchronize
    stream, ret = acl.rt.create_stream()
    check_acl(ret, "step 11 acl.rt.create_stream")
    check_acl(
        acl.rt.memset(send_aclshmem_buffer.addr, g_malloc_size, 0, g_malloc_size),
        "step 11 initialize send buffer",
    )
    check_acl(
        acl.rt.memset(recv_aclshmem_buffer.addr, g_malloc_size, 0, g_malloc_size),
        "step 11 initialize receive buffer",
    )
    ash._pyshmem.aclshmem_int32_p(send_aclshmem_buffer.addr, g_value * pe, pe)
    torch.npu.synchronize()
    dist.barrier()
    core.put(recv_aclshmem_buffer, send_aclshmem_buffer, next_pe, stream)
    core.quiet(stream)
    check_acl(acl.rt.synchronize_stream(stream), "step 11 acl.rt.synchronize_stream")
    check_acl(acl.rt.destroy_stream(stream), "step 11 acl.rt.destroy_stream")
    dist.barrier()
    recv_val = _read_int32_from_device(recv_aclshmem_buffer.addr)
    assert recv_val == g_value * prev_pe, \
        f"[FAIL] step 11: expected recv buffer {g_value * prev_pe} from PE {prev_pe}, got {recv_val}"

    # 12. free and finialize
    core.free(send_aclshmem_buffer)
    core.free(recv_aclshmem_buffer)
    core.free(signal_aclshmem_buffer)
    core.finalize()


if __name__ == "__main__":
    local_pe = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_pe)

    dist.init_process_group(backend="gloo", init_method="env://")
    run_put_signal_test()
    print("test_rma running success!")
