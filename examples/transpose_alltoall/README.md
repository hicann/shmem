# TransposeAllToAll 示例（Ascend950）

## 样例介绍

本样例展示了面向 Ascend950 的融合算子：**本地 4D 转置（BNSD → BSND）+ B 轴
AllToAll**。

## 计算定义

每个 Rank 输入 BNSD 布局的张量，形状 `[B, N, S, D]`，数据类型支持
`float16_t` / `float32_t` / `int32_t`（默认 `float16_t`），执行：

1. 本地转置 BNSD → BSND（交换 N、S 轴），得到 `[B, S, N, D]`；
2. 沿 B 轴做 AllToAll。

每个 Rank 的输出形状为 `[B, S, N, D]`（BSND），内容为全部 Rank 对应 B 分片转置后的
拼接。要求 `B % rankSize == 0`。

## 代码结构

```
transpose_alltoall/
├── CMakeLists.txt
├── main.cpp
├── transpose_alltoall_device.h
├── transpose_alltoall_kernel.hpp
└── scripts/
    ├── gen_data.py
    ├── verify_results.py
    └── run.sh
```

kernel 全部随本样例自带：转置与融合调度在 `transpose_alltoall_kernel.hpp` /
`transpose_alltoall_device.h` 中实现，跨 Rank 数据搬运直接基于 shmem 设备侧接口
（`aclshmemx_mte_get_nbi`、`shmem_ptr`、`aclshmem_signal_wait_until`、
`aclshmemx_signal_op`、`aclshmem_fence` 等）实现，不依赖 catccos/catlass。

## 构建与运行

前置条件：SOC 为 Ascend950。本样例自包含，无需额外的第三方源码树。

```bash
# 1. 构建 shmem（含 examples，Ascend950）
bash scripts/build.sh -examples -soc_type Ascend950

# 2. 正确性运行（设备列表决定 rankSize，如 "0,1" 表示 2 Rank）
cd examples/transpose_alltoall
bash scripts/run.sh "0,1"

# 3. 指定数据类型（可选，默认 float16_t）
bash scripts/run.sh -type float16_t "0,1"

# 4. 详细输出（打印 input/golden/actual 张量）
bash scripts/run.sh "0,1" -v

# 5. 跳过 golden 校验（只跑 kernel，不调用 verify_results.py）
bash scripts/run.sh --skip-check "0,1"   # 等价写法：-nc / --no-check
```

## 约束

启动前由 host 侧校验（`main.cpp` 的 `CheckTiling`）：

- 批维度 `B` 必须能被 Rank 数量 `rankSize` 整除。
- 输入侧读取步长约束：`S × D × sizeof(dtype)` 不能超过 `UINT32_MAX`（约 4 GB）。
  device 侧输入 `DataCopyExtParams` 的 GM 侧 stride 以 32 位字节数持有，`S` 与 `D`
  的乘积过大会窄化截断、导致跨 `S` 步长错误，host 侧 `CheckTiling` 会拒绝此类形状。
- 通信调度块数约束：总块数 `totalBlocks = ceil(chunkSize / commSizeM) × rankSize`
  不能超过 `INT32_MAX`。核内 `totalSteps` 以 `int32_t` 持有（与 int32 信号协议对齐），
  超出会窄化截断，host 侧 `CheckTiling`会拒绝此类形状。
- IPC 工作区大小（`WORKSPACE_STAGES` × `AICORE_NUM` × `BLOCK_SIZE_BYTES`）不能超过对称内存缓冲区总大小的一半。
