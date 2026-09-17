# 样例介绍

本样例使用 RDMA atomic 接口进行远端原子加（FAA）和比较交换（CAS）。每个 PE 对下一个 PE 执行一次 FAA，以及比较成功和比较失败两种 CAS 操作，并校验返回值和目标值。

每个 PE 使用单个 AIV 顺序调用 RDMA atomic 接口，调用期间不并发使用同一目标 PE 的 QP0。

## 环境要求

- 支持 Ascend950 平台的云脉网卡（XSCALE）和 1825 网卡（HNS_1825）。
- 至少使用两张 NPU 卡，RDMA 网卡及驱动已正确安装配置。
- CANN 版本要求、RDMA 环境检查和网卡插件库配置参见 [RDMA 样例说明](../rdma_demo/README.md#环境要求)。

## 使用方式

### 编译

参考 [RDMA 样例的编译说明](../rdma_demo/README.md#编译)，选择 Ascend950 平台下对应网卡的编译命令。

### 运行

运行前按 [RDMA 样例的环境变量说明](../rdma_demo/README.md#ibv_extend_drivers-环境变量) 设置 `IBV_EXTEND_DRIVERS`。手动运行时，按 [RDMA 样例的运行说明](../rdma_demo/README.md#运行) 配置 `LD_LIBRARY_PATH`。

#### 单机运行

在 `shmem/` 目录执行：

```bash
bash examples/rdma_atomic_demo/run.sh
```

`run.sh` 默认启动 2 个 PE，使用本机设备 0、1。可通过 `-pes` 指定 PE 数量：

```bash
bash examples/rdma_atomic_demo/run.sh -pes 4
```

#### 跨机运行

两台机器各使用一张 NPU 卡。假设机器 A 的 IP 为 `ip1`，在两台机器的 `shmem/` 目录设置：

```bash
export SHMEM_UID_SESSION_ID=ip1:8899
```

在机器 A 启动 PE0：

```bash
./build/bin/rdma_atomic_demo 2 0 tcp://ip1:8899 1 0 0
```

同时，在机器 B 启动 PE1：

```bash
./build/bin/rdma_atomic_demo 2 1 tcp://ip1:8899 1 1 0
```

#### 运行结果

单机或跨机双 PE 运行成功后，输出如下：

```text
pe=0 peer=1 faa_old=11 cas_old=21 cas_miss_old=31 target=30
[SUCCESS] demo run success in relative pe 0
pe=1 peer=0 faa_old=10 cas_old=20 cas_miss_old=30 target=31
[SUCCESS] demo run success in relative pe 1
```

### 命令行参数说明

```text
./rdma_atomic_demo <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu>
```

- `n_pes`：全局 PE 数量。
- `pe_id`：当前进程的 PE 编号。
- `ipport`：SHMEM 初始化所需的 IP 和端口，格式为 `tcp://<IP>:<端口号>`。跨机运行时使用 PE0 所在 Host 的 IP。
- `g_npus`：本机启动的 NPU 卡数量。
- `f_pe`：本机第一个 PE 的编号。
- `f_npu`：本机使用的第一张 NPU 卡的 Runtime 可见设备编号。
