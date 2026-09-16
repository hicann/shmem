# Sample Introduction

## Example Scenario

The ACLGraph structure is as follows:
![image.png](https://raw.gitcode.com/user-images/assets/8546182/b9370686-7b23-4b69-b18e-606167315795/image.png 'image.png')
Because `rdma allgather` issues only write operations, the `aclshmemx_roce_barrier_all` API is added to the `rdma allgather` operator for synchronization. This ensures that the graphs on all PEs have been executed to the corresponding stage, ensuring the precision of the ACLGraph (model).

## Environment Requirements

The environment requirements are the same as those of [rdma_demo](../rdma_demo/README_en.md).

## Instructions

1. Build in the `shmem/` directory.
    For RDMA build parameters (A2/A3, and the `XSCALE` / `HNS_1825` backends of Ascend950), see [Compilation and Build - RDMA Parameters](../../docs/compilation_build_guide_en.md#rdma-parameters).

2. Run in the `shmem/rdma_aclgraph_demo` directory.
    > Note: On the Ascend950 platform, the `IBV_EXTEND_DRIVERS` environment variable must be set. See [Environment Variable Description](../rdma_demo/README_en.md#ibv_extend_drivers-environment-variable).

    ```bash
    bash run.sh # Single-server dual-device example
    ```

3. For a dual-server 8-device example, modify the corresponding `run.sh` file.

    ```bash
    # Server 1
    pids=()
    msprof --application="./build/bin/rdma_aclgraph_demo 8 0 tcp://{IP address of server 1}:{Port number} 4 0 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 0
    pid=$!
    pids+=("$pid")

    msprof --application="./build/bin/rdma_aclgraph_demo 8 1 tcp://{IP address of server 1}:{Port number} 4 0 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 1
    pid=$!
    pids+=("$pid")

    msprof --application="./build/bin/rdma_aclgraph_demo 8 2 tcp://{IP address of server 1}:{Port number} 4 0 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 2
    pid=$!
    pids+=("$pid")

    msprof --application="./build/bin/rdma_aclgraph_demo 8 3 tcp://{IP address of server 1}:{Port number} 4 0 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 3
    pid=$!
    pids+=("$pid")

    # Server 2
    pids=()
    msprof --application="./build/bin/rdma_aclgraph_demo 8 4 tcp://{IP address of server 1}:{Port number} 4 4 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 4
    pid=$!
    pids+=("$pid")

    msprof --application="./build/bin/rdma_aclgraph_demo 8 5 tcp://{IP address of server 1}:{Port number} 4 4 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 5
    pid=$!
    pids+=("$pid")

    msprof --application="./build/bin/rdma_aclgraph_demo 8 6 tcp://{IP address of server 1}:{Port number} 4 4 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 6
    pid=$!
    pids+=("$pid")

    msprof --application="./build/bin/rdma_aclgraph_demo 8 7 tcp://{IP address of server 1}:{Port number} 4 4 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 7
    pid=$!
    pids+=("$pid")
    ```

4. Command line parameters

    ```bash
    ./rdma_aclgraph_demo <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu>
    ```

    - n_pes: number of global PEs.
    - pe_id: PE ID of the current process.
    - ipport: IP address and port number required for SHMEM initialization, in the format `tcp://<IP_address>:<port_number>`. To perform a cross-server test, set the IP address to the IP address of the host where PE0 is located.
    - g_npus: number of NPUs started on the current server.
    - f_pe: ID of the first PE used on the current server.
    - f_npu: ID of the first NPU used on the current server.
