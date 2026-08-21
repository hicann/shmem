# rdma_perftest

## 概述

`rdma_perftest` 用于测试 shmem RDMA 低阶接口（`aclshmemx_roce_put_nbi` / `aclshmemx_roce_get_nbi`）在不同数据量下的传输性能，通过 [SHMEMI_PROF_START/END](../../../src/device/utils/prof/shmemi_prof.h) 宏采集性能数据，支持带宽（`bw`）和接口延迟（`lat`）两种指标。

> 该测试结果仅做参考，性能以实际场景为准。

## 测试模式

| 模式 | 说明 |
|------|------|
| `put` | 单向 Put：PE0 调用 RDMA put 将数据传输到 PE1 |
| `bi_put` | 双向 Put：两个 PE 同时互相 put |
| `get` | 单向 Get：PE0 调用 RDMA get 从 PE1 拉取数据 |
| `bi_get` | 双向 Get：两个 PE 同时互相 get |

## 与 mte_perftest、udma_perftest 的差异

| 维度 | `mte_perftest` | `udma_perftest` | `rdma_perftest` |
|------|---------------|-----------------|-----------------|
| 引擎 | 默认 MTE | 显式 `ACLSHMEM_DATA_OP_UDMA` | RDMA 引擎 |
| 并发能力 | 同 peer 多核（默认 32 核切分数据） | 强制单核（UDMA 不允许同 peer 并发） | 单 QP；XSCALE 支持多 QP 并行，满足聚合条件时按 QP 聚合提交 |
| `-b/--block-size` | 控制核数 | 兼容入参，强制 1 | 兼容入参，实际由测试模式决定 |
| UB 缓冲 | MTE 必需 | UDMA 必须 | 必须，至少 192B，默认 192B |
| 测试模式 | put / bi_put / get / bi_get | put / bi_put / get / bi_get / **put_signal** | put / bi_put / get / bi_get |
| SOC 限制 | 通用 | 仅 Ascend950 | Ascend950（需 XSCALE/HNS_1825 后端）或 A2/A3 |
| CSV 文件名 | `<test>_<dtype>_<pe>.csv` | `udma_<test>_<dtype>_<pe>.csv` | `rdma_<metric>_<test>_<dtype>_<pe>.csv` |

## 环境要求

同 [rdma_demo](../../rdma_demo/README.md) 中的环境要求。

## 编译说明

RDMA 功能需在编译时启用 `-enable_rdma` 参数，并根据 SOC 类型配置后端。详见 [编译与构建 - RDMA 参数使用说明](../../../docs/compilation_build_guide.md#rdma参数使用说明)。

### RDMA 后端能力

下表描述 RDMA 后端接口本身是否具备对应能力，不表示 `rdma_perftest` 已经覆盖该能力。

| 后端能力 | IN_DIE | XSCALE | HNS_1825 |
|----------|--------|--------|----------|
| 适用平台 | A2/A3 | Ascend950（云脉网卡） | Ascend950（1825 网卡） |
| 基础 Put/Get NBI | 支持 | 支持 | 支持 |
| 基础 Quiet 和同步 | 支持 | 支持 | 支持 |
| 普通单 QP 聚合功能 | 不支持 | 支持 | 不支持 |
| 多 QP 功能 | 不支持 | 支持 | 不支持 |
| 多 QP 聚合功能 | 不支持 | 支持 | 不支持 |
| RDMA 原子操作 | 不支持 | 支持 | 不支持 |

### rdma_perftest 支持范围

下表描述本测试在各后端实际开放的功能。后端具备能力但本测试未覆盖时，仍标记为“不支持”。

| 本测试功能 | IN_DIE | XSCALE | HNS_1825 |
|------------|--------|--------|----------|
| Put/Get 与单向/双向模式 | 支持 | 支持 | 支持 |
| 单 QP 带宽测试 | 支持 | 支持 | 支持 |
| 单 QP 时延测试 | 支持 | 支持 | 支持 |
| 带宽测试的 `batch` 分组与完成等待 | 支持 | 支持 | 支持 |
| 显式 QP 数量和 QP 编号 | 不支持，`run.sh` 回退到默认单 QP | 支持，QP 数量范围 1~32 | 不支持，`run.sh` 回退到默认单 QP |
| 多 QP 并行带宽测试 | 不支持 | 支持，QP 数量范围 2~32 | 不支持 |
| 固定 QP 诊断模式 | 不支持 | 支持 | 不支持 |
| 聚合提交 | 不支持 | 支持，仅 `bw`、`qp_num <= 2` 且消息小于 64 KiB | 不支持 |
| 原子操作测试 | 不支持 | 不支持，未提供测试模式 | 不支持 |

## 使用方法

```bash
cd examples/shmem_perftest/rdma_perftest/
bash run.sh [选项]
```

> Ascend950 平台需设置 `IBV_EXTEND_DRIVERS` 环境变量，参见 [环境变量说明](../../rdma_demo/README.md#ibv_extend_drivers-环境变量)。

### 命令行参数

| 参数 | 缩写 | 描述 | 默认值 |
|------|------|------|--------|
| `--test-type <type>` | `-t` | 测试类型：`put` / `bi_put` / `get` / `bi_get` / `all` | `put` |
| `--datatype <type>` | `-d` | 数据类型：`float` / `int8` / `int16` / `int32` / `int64` / `uint8` / `uint16` / `uint32` / `uint64` / `char` / `all` | `float` |
| `--exponent <n>` | `-e` | 数据量幂数（2^n 字节） | - |
| `--exponent-range <min> <max>` | - | 数据量幂数范围 | `3 17` |
| `--loop-count <n>` | - | 循环次数 | `1000` |
| `--metric <bw\|lat>` | - | 性能指标：`bw`=带宽，`lat`=接口延迟 | `bw` |
| `--ub-size <n>` | - | UB 大小（B），自动 64B 对齐，范围 192~131136 | `192` |
| `--batch <n>` | - | 连续发起多少次操作后等待完成（详见下方说明） | `0` |
| `-q/--qp/--qp-count <n>` | `-q` | QP 数量，即两个 PE 之间的队列对数，范围 1~32 | `1` |
| `-i/--qp-index <n>` | `-i` | 固定使用的 QP 编号；`-1` 表示多 QP 并行模式自动分配 | `-1` |
| `--sync-id <id>` | - | 显式传给 Put/Get/Quiet 的同步 ID | `0` |
| `-b/--block-size <n>` | `-b` | 兼容入参，实际由测试模式决定 | `1` |
| `--block-range <min> <max>` | - | 兼容入参，同上 | `1 1` |
| `-pes <n>` | - | PE 数量（强制为 2） | `2` |
| `-ipport <ip:port>` | - | 通信地址 | `tcp://127.0.0.1:8768` |
| `-gnpus <n>` | - | NPU 数量 | `2` |
| `-fnpu <id>` | - | 首个 NPU ID | `0` |
| `-a/--analyse <mode>` | - | 分析模式：`none` / `plot` / `md` | `none` |

### HBM 与对称内存约束

- 仅测试 HBM（`DEVICE_SIDE`）内存路径，**不支持 D2H / `HOST_SIDE` (DRAM)**。
- 通过 `aclshmem_malloc` 分配输入/输出缓冲区，Put/Get 的本地和远端操作数都必须指向对称内存。
- 默认 1 GB 本地内存，数据量较大时自动上调（最多 40 GB）。

### 关键参数说明

#### 什么是 QP

QP（Queue Pair，队列对）是 RDMA 通信的基本单元，每个 QP 维护独立的发送队列和接收队列。`-q/--qp/--qp-count` 指定两个 PE 之间建立的 QP 数量（范围 1~32），`-i/--qp-index` 指定测试时使用哪个 QP。

#### 单 QP 模式（默认，不传 `-q` 和 `-i`）

不传任何 QP 参数时，两个 PE 之间使用 1 个 QP 完成所有传输。每次测试 `loop-count` 次数据传输，`DataSize/B` 是每次传输的数据量，带宽 = 总数据量 / 总耗时。

在 XSCALE 后端执行带宽测试且消息小于 `64 * 1024B` 时，此模式会自动将同组 NBI 操作批量聚合提交，减少提交开销。时延测试不聚合。`run.sh` 会检查 `--ub-size` 是否满足聚合所需最小值，不足时会警告并自动修正。

#### 多 QP 并行模式（`--qp N --qp-index -1`）

使用 N 个 QP 并行传输，提升总吞吐。每个 QP 使用独立的数据区，各自完成 `loop-count` 次传输。`DataSize/B` 是**单个 QP** 每次传输的数据量，每轮测试的总数据量为 `DataSize × N`。带宽按所有 QP 完成这些数据的共同时间窗口计算。

- 此模式需配合 `--metric bw` 使用。`-q N` 指定 QP 数量（范围 1~32），`-i -1` 表示自动为每个 QP 分配独立数据区。
- 例如 `-q 4 -i -1`、`DataSize=4MB`、`loop-count=1000`：4 个 QP 各自提交 1000 次 4MB 传输，单次并发总数据量 16MB，累计总数据量 16GB。
- 在 XSCALE 后端，仅 `N <= 2` 且单个 QP 的消息小于 `64 * 1024B` 时，每个 QP 会将同组 NBI 聚合后提交。`N > 2` 或消息达到该阈值时使用立即提交路径。`run.sh` 会在聚合所需的 `--ub-size` 不足时警告并自动修正。

> 仅 **云脉（XSCALE）** 支持多 QP。非云脉环境（如 HNS_1825）不支持，`run.sh` 会拦截并回退为单 QP 模式。

#### 单 QP 诊断模式（`--qp N --qp-index K`）

固定使用第 K 个 QP（0 ≤ K < N）完成全部测试，其他 QP 不参与。用于比较单个 QP 的行为或排查特定 QP 的问题。

- `-q N` 声明 QP 池大小，`-i K` 固定选择第 K 个 QP。`-i` 必须满足 `-1 ≤ K < N`，否则 `run.sh` 会报错退出。
- 时延测试（`--metric lat`）要求 `qp_num=1`，不支持 `qp_num>1`；计时只覆盖唯一 QP 的传输耗时。
- XSCALE 中，默认单 QP、固定 QP 和多 QP 模式遵循同一聚合条件：`--metric bw`、`qp_num <= 2` 且消息小于 `64 * 1024B`。其他情况使用立即提交路径。

> 仅 **云脉（XSCALE）** 支持。非云脉环境（如 HNS_1825）不支持，`run.sh` 会拦截并回退为单 QP 模式。

#### 批量与完成语义（`--batch`）

`--batch N` 控制带宽路径（`--metric bw`）中操作的分组方式：连续发起 N 次非阻塞操作后，等待这一组全部完成，再开始下一组。

- 完成语义：每组操作完成后，**Put 的源缓冲区可安全复用**（数据已被远端接收），**Get 的目标缓冲区数据已就绪可读**。
- 直接执行二进制时，`batch=0` 或 `batch > loop_count` 按 `loop_count` 处理，即所有操作完成后统一等待一次。
- 延迟路径（`--metric lat`）不按 `batch` 分组，全部 `loop-count` 次提交后统一等待。
- 对 XSCALE 聚合候选测试，`batch=0` 会被 `run.sh` 自动设为 100；`batch >= 1024` 或 `batch > loop-count` 会被修正为不超过 `loop-count` 的默认值 100。非聚合测试不执行此项修正。

#### XSCALE 平台约束

`run.sh` 通过 `IBV_EXTEND_DRIVERS` 环境变量或 `ibv_devinfo` 输出自动识别 XSCALE 环境。识别后，会对 XSCALE 聚合候选路径执行以下约束：

1. **启用条件**：仅带宽测试、`qp_num <= 2` 且测试数据范围包含小于 `64 * 1024B` 的消息时，脚本执行以下聚合准备。时延测试、`qp_num > 2` 或全部消息均不小于 64 KiB 时不聚合。
2. **UB 大小**：所需最小 UB 为 `64 + 128 × max_aggregate_count` 字节，`max_aggregate_count` 取 warmup(100) 和 `batch` 的较大值。若当前 `--ub-size` 不足则警告并自动修正，上限 131136B。
3. **batch 值**：`--batch 0` 自动设为 100；`--batch >= 1024` 或 `--batch > loop-count` 会修正为不超过 `loop-count` 的默认值 100。
4. **逐数据量路径**：满足上述测试条件时，小于 `64 * 1024B` 的消息走聚合提交，达到或超过阈值的消息走立即提交。

以上 UB 和 batch 自动调整由 `run.sh` 完成。直接执行 `build/bin/rdma_perftest` 时不会经过脚本修正；如果运行满足 XSCALE 聚合条件的带宽测试（包括固定 QP），请手动将 `--ub-size` 设置为满足 `64 + 128 × max_aggregate_count` 的容量，否则可能出现 UB 空间不足。

### 使用示例

```bash
# 默认兼容模式：单向 PUT 带宽测试
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000

# 默认兼容模式：延迟测试
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000 --metric lat

# 批量完成：每 100 次操作后等待完成，再开始下一组
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000 --batch 100

# 多 QP 带宽测试：2 个 QP 并行，每个 QP 每 100 次完成一批；小于 64KiB 时批内聚合
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000 -q 2 -i -1 --metric bw --batch 100

# 单 QP 诊断模式：仅测试编号为 0 的 QP，每 100 次操作后等待完成
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000 -q 4 -i 0 --metric bw --batch 100

# 双向 GET
./run.sh -t bi_get -d int32 --exponent-range 8 20 --loop-count 1000
```

## CSV 输出

CSV 文件输出到 `output/` 目录，命名格式为 `rdma_<metric>_<test_type>_<dtype>_<pe>.csv`。

| 列 | 说明 |
|----|------|
| `DataSize/B` | 单个 QP 每次传输的数据量（字节）；多 QP 并行模式总数据量为 `DataSize × qp-count` |
| `Npus` | NPU 数量 |
| `Blocks` | 多 QP 并行模式为 `qp-count`，其他模式为 1 |
| `UBsize/KB` | UB 大小（KB） |
| `Bandwidth/GB/s` | 带宽（GB/s，十进制） |
| `Bandwidth/GiB/s` | 带宽（GiB/s，二进制） |
| `CoreMaxTime/us` | 单次 iteration 耗时（us） |

## 已知约束

1. **并发限制**：RDMA 规范不允许同一 PE 的并发 RMA/AMO 操作。多 QP 并行模式使用独立 QP 规避此限制；单 QP 诊断模式和时延测试仅使用单 QP。
2. **编译依赖**：需启用 `-enable_rdma`；Ascend950 平台需额外指定 `-soc_type Ascend950` 及 `-rdma_backend XSCALE`（或 `HNS_1825`）。
3. **不支持 D2H**：RDMA 引擎仅测 HBM，不支持 Host 侧 DRAM。
4. **不支持原子操作**。
5. **不支持 put_signal**：RDMA 引擎无 `aclshmemx_roce_put_signal_nbi` 接口。
