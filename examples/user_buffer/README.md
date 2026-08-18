# 用户 Buffer 示例

本示例展示 `aclshmemx_init_attr_with_buffers` 的多 buffer 用法：调用方创建并映射两块独立 Device buffer，
按 descriptor 数组顺序将它们作为对称堆前缀交给 SHMEM，再从 SHMEM 管理的 tail 分配源数据，通过 MTE 或
UDMA 分别向相邻 PE 的两块用户 buffer 执行 put。

两块源数据使用不同的 PE 和 buffer 特征值。通信完成后，每个 PE 将两块用户 buffer 分别拷回 Host，并对
64 KiB payload 做逐字节精确比较；任一 buffer 出现不匹配时样例返回失败。单元测试继续覆盖仅 VA 输入、
buffer 内偏移和越界查询、单/双 PE、tail 分配以及资源释放等组合。

## 编译

```bash
source /path/to/cann/set_env.sh
bash scripts/build.sh -examples
```

UDMA 需要 Ascend950 构建：

```bash
bash scripts/build.sh -examples -soc_type Ascend950
```

产物为 `build/bin/user_buffer` 和 `build/lib/libuser_buffer_kernel.so`。

## 运行

默认在 NPU 0、1 上运行 MTE：

```bash
bash examples/user_buffer/run.sh
```

在 Ascend950 上运行 UDMA：

```bash
bash examples/user_buffer/run.sh -engine udma
```

也可通过 `-pes`、`-gnpus`、`-fnpu` 和 `-ipport` 调整 PE、设备和控制面地址。成功时每个 PE 输出：

```text
[SUCCESS] PE 0 verified 2 caller-owned buffers with mte
```

用户创建的两组 source VA mapping 和 physical handle 始终由用户持有。示例在 `aclshmem_finalize` 返回后按
逆序解除映射并释放 handle；SHMEM-owned tail 由 SHMEM 自行释放。
