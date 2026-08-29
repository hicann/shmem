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
import torch
import torch.distributed as dist
import shmem as ash
import shmem.core as core


g_ash_size = 1024 * 1024 * 1024
g_malloc_size = 8 * 1024 * 1024


def run_direct_test():
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

    # 2. init with unique id
    core.init(rank=pe, nranks=world_size, mem_size=g_ash_size, uid=unique_id, initializer_method='uid')

    # 3. get init_status
    status = core.direct.init_status()
    if status != core.direct.InitStatus.INITIALIZED:
        raise ValueError('[ERROR] init_status failed')

    # 4. my_pe, n_pes
    my_pe, pe_count = core.direct.my_pe(), core.direct.n_pes()
    if not (my_pe == pe and pe_count == world_size):
        raise ValueError('[ERROR] pe/world failed')

    # Invalid split parameters must raise instead of being confused with the
    # valid ACLSHMEM_TEAM_INVALID result returned for a non-member PE.
    try:
        ash.team_split_strided(0, -1, 1, 1)
    except RuntimeError:
        pass
    else:
        raise ValueError('[ERROR] invalid team split did not raise RuntimeError')

    non_member_team = ash.team_split_strided(0, 0, 1, 1)
    if pe == 0:
        if non_member_team < 0:
            raise ValueError('[ERROR] member PE did not receive a valid team')
        ash.team_destroy(non_member_team)
    elif non_member_team != core.ACLSHMEM_TEAM_INVALID:
        raise ValueError('[ERROR] non-member PE did not receive ACLSHMEM_TEAM_INVALID')

    # 5. create team
    team_id = ash.team_split_strided(0, pe, 1, 1)

    # 6. team_my_pe, team_n_pes
    team_my_pe = core.direct.team_my_pe(team_id)
    if team_my_pe != 0:
        raise ValueError('[ERROR] team_my_pe failed')
    team_n_pes = core.direct.team_n_pes(team_id)
    if team_n_pes != 1:
        raise ValueError('[ERROR] team_n_pes failed')

    # 7. finialize
    core.finalize()


if __name__ == "__main__":
    local_pe = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_pe)

    dist.init_process_group(backend="gloo", init_method="env://")
    run_direct_test()
    print("test_direct running success!")
