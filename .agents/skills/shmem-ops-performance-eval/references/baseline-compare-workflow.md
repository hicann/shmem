# Baseline 接入与性能对比工作流

本文规定 `shmem-ops-performance-eval` 在采集 SHMEM 算子性能时，**如何接入 HCCL/aclnn baseline、如何跑对比、如何在聊天与报告中输出对比结果**。以 `custom-ops/alltoallv` 为参考实现。

**命令代码段唯一参考**：[perf-workflow.md](perf-workflow.md)（Skill 内禁止引用独立 shell 脚本文件名）。

## 1. 核心要求

- 有 HCCL/aclnn baseline 时 **MUST** 实现 C++ baseline 可执行文件，**NEVER** 仅用 Python/torch 脚本代替
- **MUST** 用与 SHMEM 相同的 case（同一 `gen_data.py` 输出、同一 sendcounts/shape/dtype/PE 数）
- **MUST** 在聊天界面输出 **SHMEM vs baseline 对比表**（不得只报 `[PERF]` 单边数据）
- **MUST** 写入 `docs/performance_report.md` §3.5 性能对比表
- 达标线：**按算子**见 [platform-perf-spec.md](platform-perf-spec.md)（非统一 80%）
- 规模：**MUST** 通过 **带宽 sweep** 找到平台区（稳定带宽），**NEVER** 用未饱和的小 payload 带宽作最终结论

### 1.1 稳定带宽（平台区）— MUST

通信算子带宽 **随 payload 增大而上升** 说明固定开销（launch、barrier、同步）仍占主导，**不是**链路饱和带宽。

**MUST** 按以下流程测 **稳定带宽**：

1. **倍增 sweep**：对 `base_count` 做 2× 递增；HCCL 与 SHMEM **分阶段采集**（见 [perf-workflow.md §2](perf-workflow.md)）
2. **双指标**：
   - `e2e_bus_bandwidth_GBps`：含 host `aclshmem_barrier_all`，小 payload 会 **严重低估**
   - `steady_bus_bandwidth_GBps`：按 **kernel_us** 计算，**对比与达标 MUST 用此指标**
3. **平台判定**：per-PE payload ≥ 64MB 且相邻 2× 点 `steady_bus` 相对变化 **≤ 5%**
4. **最终报告**：只取平台区 `steady_bus` 写入对比表；sweep 全表 **MUST** 附在聊天/报告 §3.1

分阶段采集（**MUST — HCCL 不得在 SHMEM init 之后同会话调用**）：按 [perf-workflow.md §1](perf-workflow.md) 执行阶段 A（baseline）→ 阶段 B（SHMEM）→ 阶段 C（离线 `perf_compare.py`）。

**NEVER** 将 base=1M（~16MB/PE fp16）等 M 档带宽当作「最终性能」。

## 2. 目录与产物（custom-ops）

```
custom-ops/
├── scripts/lib/perf_compare.py       # 带宽离线对比
├── scripts/perf_compare_latency.py   # 时延离线对比（可选）
└── <op_name>/
    ├── baseline/scripts/run_baseline   # HCCL baseline 入口
    ├── scripts/perf                  # SHMEM perf 入口
    └── data/perf/                    # baseline_*.log / shmem_*.log
```

### 2.0 HCCL 与 SHMEM 隔离（Hard Gate）

| 规则 | 说明 |
| --- | --- |
| **NEVER 同会话混跑** | 禁止在同一 shell / 同一 `docker exec` 内先跑 SHMEM 再跑 HCCL（或反之） |
| **分阶段采集** | 阶段 A 与阶段 B **独立执行**（见 perf-workflow.md） |
| **离线对比** | 阶段 C **只读日志**，不启动算子进程 |
| **推荐** | 两次独立 `docker exec`，中间间隔 ≥30s |
| **禁止** | 同一脚本内连续启动 HCCL + SHMEM 进程 |

### 2.1 CMake 集成

算子根 `CMakeLists.txt` 末尾：

```cmake
add_subdirectory(baseline)
```

baseline 的 `CMakeLists.txt` 将 `CMAKE_RUNTIME_OUTPUT_DIRECTORY` 设为 `${CMAKE_BINARY_DIR}/../bin`，与算子主程序同目录。

### 2.2 日志标签约定

| 实现 | 输出前缀 | 必含字段 |
| --- | --- | --- |
| SHMEM 算子 | `[PERF]` | `e2e_us`, `kernel_us`, `bus_bandwidth_GBps`, **`steady_bus_bandwidth_GBps`**, `payload_bytes` |
| HCCL/aclnn baseline | `[BASELINE_PERF]` | 同上；额外 `api=HcclAlltoAllV` 等 |

- **`steady_bus_bandwidth_GBps`** = `payload_bytes / kernel_us * bus_factor / 1e9` — **达标对比 MUST 用此字段**
- **`bus_bandwidth_GBps`**（e2e 口径）仅作参考

对比通过 grep 上述两行 + `perf_compare.py` 解析，**禁止**手工抄数。

## 3. 分阶段性能采集与离线对比

### 3.1 阶段职责

| 阶段 | 启动进程 | 输出 |
| --- | --- | --- |
| A — baseline | 仅 HCCL | `data/perf/baseline_*.log` + `[BASELINE_PERF]` |
| B — SHMEM | 仅 SHMEM 算子 | `data/perf/shmem_*.log` + `[PERF]` |
| C — compare | **无**（离线 python） | 对比表 |

命令代码段见 [perf-workflow.md §1](perf-workflow.md)；sweep 见 §2。

### 3.2 工作流

1. 阶段 A → 等待 ≥30s → 阶段 B → 阶段 C
2. Agent **MUST** 将对比表摘要到聊天

### 3.3 Docker 参考

每阶段独立 `docker exec`，容器内粘贴 [perf-workflow.md](perf-workflow.md) 对应代码段（见 [docker-exec-contract.md](../../shmem-ops-dev/references/docker-exec-contract.md)）。

## 4. 聊天界面输出模板（MUST — 自动触发）

完整规范见 [perf-chat-output-spec.md](perf-chat-output-spec.md)。

### Phase 6 最小示例

```text
### <case_id> — <scale> <dtype> 8PE

| 指标 | SHMEM | HCCL baseline | 比值 | 目标 | 达标 |
|------|-------|---------------|------|------|------|
| steady_bus (GB/s) | ... | ... | ...% | 见 platform-perf-spec | PASS/FAIL |
| kernel (μs) | ... | ... | ...× | — | — |
```

**NEVER** 只输出 SHMEM 单边 `[PERF]`。

## 5. AllToAllV baseline 选型（参考）

1. 优先 `HcclAlltoAllV` / `HcclAlltoAllVC`
2. baseline 复用算子 `op_host_plan.cpp` 解析 `sendcounts.bin`
3. HCCL comm：`HcclGetRootInfo` + 文件广播 + `HcclCommInitRootInfo`

## 6. performance_report.md 必填对比节

§3.5 **MUST** 含 steady_bus 对比；§7 **MUST** 按 [platform-perf-spec.md](platform-perf-spec.md) 说明达标/未达标原因。

## 7. alltoallv 实测参考（910B3 8卡 sweep）

| base_count | payload MB/PE | SHMEM steady_bus | HCCL steady_bus | 平台? |
| --- | --- | --- | --- | --- |
| 262144 | 4.2 | ~14.5 | ~9.3 | 否 |
| 1048576 | 16.8 | ~36.1 | ~10.5 | 否 |
| 4194304 | 67.1 | ~68.2 | ~11.1 | HCCL 是 |
| 16777216 | 268.4 | ~83.2 | ~10.8 | HCCL 是 |

交付 **MUST** 跑 sweep（perf-workflow §2），不得只报单点 M 档 e2e 带宽。

## 8. 反模式

- ❌ 同一 shell / docker exec 内先 SHMEM 再 HCCL
- ❌ 同一流程内嵌启动 baseline + shmem
- ❌ baseline 用 Python dist 代替 C++ HCCL
- ❌ 对比表只写文件路径，聊天无表格
- ❌ 在宿主机无 NPU 环境伪造 perf 数据
