# 平台规格性能表（Phase 6 强制输出）

Agent 在 Phase 6 完成后 **MUST** 在聊天输出与下表对齐的对比结果（数值来自 `[PERF]` / `[BASELINE_PERF]`，**NEVER** 编造）。

环境：单机 8PE、full-mesh、P2P 28GB/s、fp16（moe 按设计 dtype）。

## 目标表

| 算子 | 规模 | 指标 | 目标 |
| --- | --- | --- | --- |
| reduce_scatter | 256MB / 8PE | SHMEM steady_bus / HCCL steady_bus | **98.66%** |
| alltoallv | 1GB / 8PE | SHMEM steady_bus / HCCL steady_bus | **105%** |
| moe_dispatch | 64KB / 8PE | HCCL 时延 / SHMEM 时延 | **4.99x** |
| allgatherv | 15.9KB / 8PE | HCCL 时延 / SHMEM 时延 | **11.19x** |

## 默认采集参数

| 算子 | 参数 |
| --- | --- |
| reduce_scatter | `elements=134217728` float16 |
| alltoallv | `uniform base_count=8388608` float16 |
| allgatherv | `variable total_elems=8150` float16 |
| moe_dispatch | `bs=4 h=64 topk=2 expert_per_pe=1` |

## 生成对比表

按 [perf-workflow.md](perf-workflow.md) §1 对四算子分别执行三阶段采集（A baseline → B SHMEM → C 离线对比），参数见上表 `ITERS=30`。

输出含「带宽表 + 时延表」；Agent **MUST** 将对比 Markdown 表贴入聊天（格式见 [perf-chat-output-spec.md](perf-chat-output-spec.md)）。

## 达标口径

- 带宽算子：比值 = SHMEM_steady_bus / HCCL_steady_bus（HCCL 未采到则标注并 retry baseline）
- 时延算子：比值 = HCCL_comm_only / SHMEM_comm_only（无 `comm_only_us` 时用 `kernel_us`）
