# 样例介绍

本样例旨在展示 SIMD 与 SIMT 混合编译模式下，SIMT 远程内存访问（RMA）接口使用 UB 作为中转缓冲区进行数据搬运的典型方法。样例代码通过 `__simt_vf__` 函数在 Device 侧申请 UB 数组，并调用 SIMT RMA NBI 接口在 GM 与 UB 之间完成数据传输。

本样例主要演示以下接口形式：

UB↔GM 类 RMA 接口与 GM↔GM 类接口的组织方式一致，同样按“数据长度指定方式”分为三种形式（以 `get` 为例，`put` 的方向相反）：

1. `__simt_callee__ inline void aclshmem_{NAME}_get_nbi(__ubuf__ TYPE *dst, __gm__ TYPE *src, size_t elem_size, int32_t pe)`
2. `__simt_callee__ inline void aclshmem_get{BITS}_nbi(__ubuf__ void *dst, __gm__ void *src, size_t nelems, int32_t pe)`
3. `__simt_callee__ inline void aclshmem_getmem_nbi(__ubuf__ void *dst, __gm__ void *src, size_t elem_size, int32_t pe)`

- **第一种接口**：基于每个传输元素的具体数据类型（如 `half`、`float` 等）进行描述。
- **第二种接口**：基于每个传输元素的比特位大小（如 `8`、`16` 等）进行描述。
- **第三种接口**：直接指定需要传输的总内存字节大小。

占位符可选值如下表所示：

| 占位符 | 可选值 |
| --- | --- |
| `{NAME}` | `half`, `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `char`, `bfloat16` |
| `{BITS}` | `8`, `16`, `32`, `64`, `128` |

每种形式又各有 3 个线程组粒度的变体，区别在于参与本次搬运的线程范围：

| 变体 | 说明 |
| --- | --- |
| `aclshmem_*`（无后缀） | 线程级，由调用线程独立完成整块数据的搬运 |
| `aclshmemx_*_block` | block 级，由同一 block 内的线程协作完成搬运 |
| `aclshmemx_*_warp` | warp 级，由同一 warp（32 个线程）协作完成搬运 |

此外，每个接口还有同步（阻塞）与异步两个版本，带 `_nbi`（non-blocking initiate）后缀的为异步版本，接口返回时数据搬运尚未完成。

**本样例调用的是异步 + warp 级的组合**，kernel 以 `dim3(32)` 启动，由 warp 内 32 个线程协作搬运同一份数据，而非每个线程各自遍历整块内存。实际调用的接口如下：

```cpp
simt::aclshmemx_int32_get_nbi_warp(__ubuf__ int32_t *dst, __gm__ int32_t *src, size_t elem_size, int32_t pe);
simt::aclshmemx_int32_put_nbi_warp(__gm__ int32_t *dst, __ubuf__ int32_t *src, size_t elem_size, int32_t pe);
```

其中：

| 参数 | 说明 |
| --- | --- |
| `dst` | 目标地址，`get_nbi_warp` 中为 UB 地址，`put_nbi_warp` 中为 GM 地址 |
| `src` | 源地址，`get_nbi_warp` 中为 GM 地址，`put_nbi_warp` 中为 UB 地址 |
| `elem_size` | 传输的 `int32_t` 元素个数 |
| `pe` | 目标或源 PE 编号 |

## 样例执行流程

本样例通过以下流程演示 UB 到 GM 的 RMA 数据路径：

1. **环境初始化**：每个 PE 初始化 3 块大小相同的对称内存。其中，`origin` 数据初始化为 `[my_pe + 0, ..., my_pe + size - 1]`，`res_prev` 和 `res_next` 初始化为 `-1`。
2. **本地 GM 到 UB**：每个 PE 使用 `aclshmemx_int32_get_nbi_warp` 将自身 `origin` 中的数据读取到 UB 缓冲区。
3. **UB 到远端 GM**：每个 PE 使用 `aclshmemx_int32_put_nbi_warp` 将 UB 缓冲区中的数据写入**上一个 PE** 的 `res_next`。之所以写向上一个 PE，是因为对它而言当前 PE 正是它的“下一个 PE”，这样每个 PE 的 `res_next` 最终存放的就是其下一个 PE 的 `origin` 数据。
4. **远端 GM 到 UB**：每个 PE 使用 `aclshmemx_int32_get_nbi_warp` 将逻辑上属于上一个 PE 的 `origin` 数据读取到 UB 缓冲区。
5. **UB 到本地 GM**：每个 PE 使用 `aclshmemx_int32_put_nbi_warp` 将 UB 缓冲区中的数据写入自身 `res_prev`。
6. **结果校验**：通信操作完成后，各 PE 将数据拷贝回 Host 并自动校验传输结果——`res_prev` 应等于上一个 PE 的 `origin`，`res_next` 应等于下一个 PE 的 `origin`。

其中步骤 2~3 演示的是「UB 中转 + 远端写」，步骤 4~5 演示的是「远端读 + UB 中转」，二者合起来覆盖了 UB 与 GM 之间的四条数据路径。

## 支持的设备

- Ascend950

## 目录结构

```text
examples/simt_rma_ub2gm/
├── CMakeLists.txt
├── README.md
├── main.cpp
└── run.sh
```

## 使用方式

1. **编译项目**

   在 `shmem/` 根目录下执行编译脚本：

   ```bash
   bash scripts/build.sh -examples -enable_simt -soc_type Ascend950
   ```

2. **运行 simt_rma_ub2gm 示例程序**

   进入示例目录并执行运行脚本：

   ```bash
   cd examples/simt_rma_ub2gm
   bash run.sh
   ```

   `run.sh` 默认启动 2 个独立进程，每个进程对应一个 PE 并绑定一张卡，使用 `build/bin/simt_rma_ub2gm` 执行样例。可通过 `-pes` 参数指定参与运行的 PE（卡）数量，例如启动 4 卡运行：

   ```bash
   bash run.sh -pes 4
   ```

   使用多卡运行时，请确保执行环境中实际可用的 NPU 卡数不少于指定的 PE 数量。

3. **查看结果**

   样例运行结束后会打印各 PE 的 `origin`、`res_prev`、`res_next` 数据摘要。若校验通过，会输出类似如下日志：

   ```text
   [SUCCESS] PE 0: Verification passed for RMA transfers.
   [SUCCESS] PE 1: Verification passed for RMA transfers.
   ```
