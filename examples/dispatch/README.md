# MoE Dispatch 示例目录

本目录聚合 dispatch 相关示例，公共数据生成和校验脚本放在父级 `scripts/`，具体算子实现放在独立子目录。

## 目录结构

```text
examples/dispatch/
├── scripts/
│   ├── data_gen.py           # classic 与 doubleplane 共用的数据生成脚本
│   └── check_dispatch.py     # classic 与 doubleplane 共用的结果校验脚本
├── dispatch_classic/
│   ├── main.cpp
│   ├── dispatch_kernel.cpp
│   └── scripts/run.sh
└── dispatch_doubleplane/
    ├── main.cpp
    ├── dispatch_doubleplane_kernel.cpp
    └── scripts/run.sh
```

## 如何使用

经典 dispatch：

```bash
cd examples/dispatch/dispatch_classic
bash scripts/run.sh -pes 2 -bs 8 -h 16 -topk 2 -expertPerPe 2 -type int32_t
```

双平面 dispatch：

```bash
cd examples/dispatch/dispatch_doubleplane
bash scripts/run.sh -pes 2 -bs 8 -h 16 -topk 2 -expertPerPe 2 -type int32_t
```

每个 `run.sh` 都会在当前 case 目录下生成 `golden/` 和 `output/`，并调用父级公共脚本 `../scripts/data_gen.py`、`../scripts/check_dispatch.py`。

## 选择建议

先使用 `dispatch_classic` 建立正确性和性能基线；当 `bs`、`h`、`topk` 较大且路由分布存在大 segment 时，再使用 `dispatch_doubleplane` 对比 `comm_only` 和 `full_op` 指标。

## 卡数限制说明

示例按 `pe_size` 启动相同数量的 AIV 并配置同数 SDMA QP，底层最多支持 72 个 AIV/QP（`ACLSHMEM_MAX_AIV_PER_NPU`）。

开启性能采集（设置环境变量 `SHMEM_CYCLE_PROF_PE`）时，`SHMEMI_PROF` 打点数组仅有 64 个 block 槽位（`ACLSHMEM_CYCLE_PROF_MAX_BLOCK`），因此请确保 `pe_size` 不超过 64；超出部分的 AIV 打点数据无法记录，可能导致性能数据不完整。不开启性能采集时不受此限制。
