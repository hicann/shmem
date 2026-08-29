# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
import threading

import torch
import torch.distributed as dist

import shmem as ash
import shmem.core as core
from shmem.core.utils import AclshmemInvalid


G_ASH_SIZE = 1024 * 1024 * 1024
G_UB_SIZE = 128


def _broadcast_unique_id(pe):
    unique_id = core.get_unique_id() if pe == 0 else None
    uid_list = [unique_id]
    dist.broadcast_object_list(uid_list, src=0)
    return uid_list[0]


def _expect_invalid(call, case_name):
    try:
        call()
    except AclshmemInvalid:
        return
    raise AssertionError(f"[FAIL] {case_name}: expected AclshmemInvalid")


def _expect_value_error(call, case_name):
    try:
        call()
    except ValueError:
        return
    raise AssertionError(f"[FAIL] {case_name}: expected ValueError")


def run_sync_config_prof_test():
    pe = dist.get_rank()
    world_size = dist.get_world_size()
    if ash.set_conf_store_tls(False, "") != 0:
        raise ValueError("[ERROR] disable TLS failed.")

    core.init(
        rank=pe,
        nranks=world_size,
        mem_size=G_ASH_SIZE,
        uid=_broadcast_unique_id(pe),
        initializer_method="uid",
    )

    # Engine-specific workspace minima are rejected before entering native code.
    _expect_invalid(
        lambda: core.set_rdma_config(0, 127, 0),
        "RDMA workspace smaller than 128 bytes",
    )
    _expect_invalid(
        lambda: core.set_udma_config(0, 127, 0),
        "UDMA workspace smaller than 128 bytes",
    )

    # Configuration APIs update the runtime state for later device-side operations.
    core.set_mte_config(0, G_UB_SIZE, 0)
    core.set_sdma_config(0, G_UB_SIZE, 0)
    core.set_rdma_config(0, G_UB_SIZE, 0)
    core.set_udma_config(0, G_UB_SIZE, 0)

    # All PEs must execute these collective calls in the same order.
    core.barrier(core.ACLSHMEM_TEAM_WORLD)
    core.barrier_all()
    core.sync(core.ACLSHMEM_TEAM_WORLD)
    core.sync_all()

    stream = int(torch.npu.current_stream().npu_stream)
    core.barrier_on_stream(core.ACLSHMEM_TEAM_WORLD, stream)
    core.barrier_all_on_stream(stream)
    torch.npu.synchronize()

    # Team-taking APIs must reject out-of-range, never-created, and destroyed
    # handles before issuing a void native synchronization call.
    never_created_team = core.ACLSHMEM_MAX_TEAMS - 1
    _expect_invalid(
        lambda: core.barrier(core.ACLSHMEM_MAX_TEAMS),
        "barrier out-of-range team",
    )
    _expect_invalid(
        lambda: core.sync(never_created_team),
        "sync never-created team",
    )
    _expect_invalid(
        lambda: core.barrier_on_stream(never_created_team, stream),
        "barrier_on_stream never-created team",
    )
    _expect_value_error(
        lambda: core.Handle(core.ACLSHMEM_MAX_TEAMS),
        "Handle out-of-range team",
    )
    _expect_value_error(
        lambda: core.Handle(never_created_team),
        "Handle never-created team",
    )
    _expect_value_error(
        lambda: ash._pyshmem.aclshmem_barrier(never_created_team),
        "low-level barrier never-created team",
    )
    world_handle = core.Handle(core.ACLSHMEM_TEAM_WORLD)
    _expect_value_error(
        lambda: setattr(world_handle, "team_id", never_created_team),
        "Handle setter never-created team",
    )

    transient_team = ash.team_split_strided(
        core.ACLSHMEM_TEAM_WORLD, 0, 1, world_size
    )
    if transient_team < 0:
        raise RuntimeError(
            f"[ERROR] create transient team failed, ret={transient_team}."
        )
    destroyed_handle = core.Handle(transient_team)
    core.barrier(transient_team)
    ash.team_destroy(transient_team)
    dist.barrier()
    _expect_invalid(
        lambda: core.barrier(transient_team),
        "barrier destroyed team",
    )
    _expect_invalid(
        lambda: core.sync(transient_team),
        "sync destroyed team",
    )
    _expect_invalid(
        lambda: core.barrier_on_stream(transient_team, stream),
        "barrier_on_stream destroyed team",
    )
    _expect_invalid(
        lambda: core.handle_wait(destroyed_handle, 0),
        "handle_wait destroyed team",
    )
    _expect_value_error(
        lambda: ash._pyshmem.aclshmemx_handle_wait(destroyed_handle, 0),
        "low-level handle_wait destroyed team",
    )
    _expect_value_error(
        lambda: core.Handle(transient_team),
        "Handle destroyed team",
    )

    prof = core.get_prof(verbose=False)
    if pe == 0:
        if not isinstance(prof, core.ProfData):
            raise ValueError("[ERROR] profiling snapshot is unavailable on configured PE 0.")
        if (
            prof.pe_id != 0
            or not isinstance(prof.ccount, list)
            or not isinstance(prof.cycles, list)
            or len(prof.ccount) != 64
            or len(prof.cycles) != 64
            or any(not isinstance(row, list) or len(row) != 1024 for row in prof.ccount)
            or any(not isinstance(row, list) or len(row) != 1024 for row in prof.cycles)
        ):
            raise ValueError("[ERROR] profiling snapshot shape or PE ID is invalid.")
        if not isinstance(prof.ccount[0][0], int) or not isinstance(prof.cycles[0][0], int):
            raise ValueError("[ERROR] profiling matrices must contain Python integers.")

        start_barrier = threading.Barrier(3)
        snapshots = []

        def get_snapshot():
            start_barrier.wait()
            snapshots.append(core.get_prof(verbose=False))

        threads = [threading.Thread(target=get_snapshot) for _ in range(2)]
        for thread in threads:
            thread.start()
        start_barrier.wait()
        for thread in threads:
            thread.join()
        if len(snapshots) != 2 or any(not isinstance(item, core.ProfData) for item in snapshots):
            raise ValueError("[ERROR] concurrent profiling snapshots are unavailable.")
        core.show_prof()
    elif prof is not None:
        raise ValueError("[ERROR] profiling data must only be returned on the configured PE.")

    core.finalize()


if __name__ == "__main__":
    local_pe = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_pe)
    dist.init_process_group(backend="gloo", init_method="env://")
    run_sync_config_prof_test()
    print("test_sync_config_prof running success!")
