# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import gc
import os
import torch
import torch.distributed as dist
import shmem as ash
import shmem.core as core
from shmem.core.utils import AclshmemInvalid, _validate_instance_id

native = ash._pyshmem


g_ash_size = 1024 * 1024 * 1024
g_malloc_size = 8 * 1024 * 1024


def _expect_invalid(call, case_name):
    try:
        call()
    except AclshmemInvalid:
        return
    raise AssertionError(f"[FAIL] {case_name}: expected AclshmemInvalid")


def _expect_native_invalid(call, case_name):
    try:
        call()
    except ValueError:
        return
    raise AssertionError(f"[FAIL] {case_name}: expected ValueError")


def run_init_final_test():
    pe = dist.get_rank()
    world_size = dist.get_world_size()
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

    # 2. The documented public instance-ID range is 0 through 254.
    _validate_instance_id(0)
    _validate_instance_id(254)
    boundary_attr = native.InitAttr()
    boundary_attr.instance_id = 0
    boundary_attr.instance_id = 254
    for invalid_id in (True, -1, 255, 1 << 64):
        _expect_invalid(
            lambda invalid_id=invalid_id: _validate_instance_id(invalid_id),
            f"instance_id={invalid_id!r}",
        )
        _expect_native_invalid(
            lambda invalid_id=invalid_id: setattr(boundary_attr, "instance_id", invalid_id),
            f"native InitAttr.instance_id={invalid_id!r}",
        )
        _expect_native_invalid(
            lambda invalid_id=invalid_id: native.aclshmemx_finalize(invalid_id),
            f"native finalize instance_id={invalid_id!r}",
        )
        _expect_native_invalid(
            lambda invalid_id=invalid_id: native.aclshmemx_instance_ctx_set(invalid_id),
            f"native instance_ctx_set instance_id={invalid_id!r}",
        )

    _expect_invalid(
        lambda: core.init(
            rank=pe,
            nranks=world_size,
            mem_size=g_ash_size,
            uid=unique_id,
            initializer_method="uid",
            instance_id=255,
        ),
        "init instance_id upper boundary",
    )
    _expect_invalid(lambda: core.finalize(instance_id=True), "finalize bool instance_id")
    _expect_invalid(lambda: core.set_instance(-1), "set_instance negative instance_id")

    def enter_invalid_instance_context():
        with core.multi_instance(255):
            pass

    _expect_invalid(enter_invalid_instance_context, "multi_instance upper boundary")

    # 3. Build InitAttr twice so the private C++ UID owner replacement path is
    # covered.  Deleting both Python variables must not invalidate comm_args.
    attr = ash.InitAttr()
    first_uid = ash.UniqueId.from_bytes(unique_id)
    ret = ash.aclshmemx_set_attr_uniqueid_args(pe, world_size, g_ash_size, first_uid, attr)
    if ret != 0:
        raise ValueError('[ERROR] first unique-ID attr setup failed')

    second_uid = ash.UniqueId.from_bytes(unique_id)
    ret = ash.aclshmemx_set_attr_uniqueid_args(pe, world_size, g_ash_size, second_uid, attr)
    if ret != 0:
        raise ValueError('[ERROR] second unique-ID attr setup failed')

    del first_uid
    del second_uid
    gc.collect()

    try:
        attr._uid_owner = object()
    except AttributeError:
        pass
    else:
        raise ValueError('[ERROR] InitAttr unexpectedly exposes writable UID owner state')

    try:
        del attr._uid_owner
    except AttributeError:
        pass
    else:
        raise ValueError('[ERROR] InitAttr unexpectedly exposes deletable UID owner state')

    ret = ash.aclshmemx_init_attr(ash.InitMode.UNIQUEID, attr)
    if ret != 0:
        raise ValueError('[ERROR] aclshmemx_init_attr failed after UID owner GC test')

    # 4. finialize
    core.finalize()


if __name__ == "__main__":
    local_pe = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_pe)

    dist.init_process_group(backend="gloo", init_method="env://")
    run_init_final_test()
    print("test_init_final running success!")
