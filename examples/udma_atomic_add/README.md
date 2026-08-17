# 样例介绍

## 版本和平台支持说明

- UDMA 原子加样例仅支持 Ascend950 平台，其他平台不支持运行该样例。
- CANN 9.1.0 已提供 UDMA 所需的 `HcommEndpointCreate`、`HcommMemReg`、`HcommChannelCreate`、`HcommChannelGetStatus` 等 HCOMM 资源接口。请从 [CANN 9.1.0 资源](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.1.0)下载并安装 Ascend950 对应的 toolkit 包和 ops 包；低于 9.1.0 的 CANN 版本不在本样例支持范围内。
- 初始化时会加载并检查所需 HCOMM 符号。编译前需执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`（自定义安装路径时使用对应的 `set_env.sh`），并使用 `-soc_type Ascend950` 编译；若 HCOMM 运行时库或符号不完整，初始化会失败。

使用方式:

1. 在shmem/目录编译:

    ```bash
    bash scripts/build.sh -examples -soc_type Ascend950
    ```

2. 在shmem/目录运行:

    ```bash
    export PROJECT_ROOT=<shmem-root-directory>
    export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
    export SHMEM_UID_SESSION_ID=127.0.0.1:8899
    ./build/bin/udma_atomic_add 2 0 tcp://127.0.0.1:8899 2 0 0 & # PE 0
    ./build/bin/udma_atomic_add 2 1 tcp://127.0.0.1:8899 2 0 0 & # PE 1
    ```

3. 命令行参数说明

    ```bash
    ./udma_atomic_add <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu>
    ```

    - n_pes: 全局PE数量。
    - pe_id: 当前进程的PE号。
    - ipport: SHMEM初始化需要的IP及端口号，格式为`tcp://<IP>:<端口号>`。
    - g_npus: 当前机器上启动的NPU卡的数量。
    - f_pe: 当前机器上使用的第一个PE号。
    - f_npu: 当前机器执行本样例使用的第一张NPU卡的卡号
