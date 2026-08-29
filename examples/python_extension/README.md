# SHMEM Python 扩展测试

本目录覆盖原有 Python 接口，以及新增的 UID 属性、多实例、``mem_type`` 对称堆、引擎配置、集合同步、profiling 和 ``handle_wait`` 接口。

## 1. 前置条件

- 已安装匹配当前 CANN、Python 和 CPU 架构的 SHMEM wheel。
- 每个本地进程可独占一张 NPU；运行前已加载 CANN 和 SHMEM 环境变量。
- 多 PE 用例通过 ``torchrun`` 启动，所有 PE 必须执行相同顺序的集合操作。

## 2. 运行全部用例

默认使用两张 NPU：

```bash
bash examples/python_extension/run.sh
```

单卡 smoke（用于绑定、单 PE 集合路径和多实例隔离检查）：

```bash
NPROC_PER_NODE=1 bash examples/python_extension/run.sh
```

若 CANN 运行时支持 Host 对称堆，可同时打开 ``HOST_SIDE`` 分配测试：

```bash
SHMEM_TEST_HOST_HEAP=1 NPROC_PER_NODE=2 bash examples/python_extension/run.sh
```

脚本默认单独用一个 PE 跑 MTE ``handle_wait`` smoke，避免在未启用 ROCE 时把该接口用于多 PE。可用 ``HANDLE_WAIT_NPROC_PER_NODE`` 覆盖，但多 PE ``handle_wait`` 应优先按下一节在真实 ROCE 环境验证。

## 3. ``putmem_on_stream`` Python/C++ 性能对比

``perf`` 目录使用同一个 SHMEM 实例、同一组对称缓冲区、相同数据长度、相同 PE ring
路由和同一个显式 ACL stream，对比：

- C++ 参考路径：辅助动态库在 C++ 循环中直接调用
  ``aclshmemx_putmem_on_stream``；
- Python 路径：Python 循环调用高层 ``shmem.core.put``，经 pybind11 进入同一个
  ``aclshmemx_putmem_on_stream``。

两条路径均在计时区间末尾同步相同 stream。测试预热后交替执行 C++/Python 路径，
以各轮所有 PE 中的最大耗时作为整组完成时间，再取多轮中位数。测试结束后抽样校验
接收缓冲区首尾字节，确保 RMA 结果正确。默认门限为 Python overhead 不超过 5%。

运行 2、4、8 PE：

```bash
NPROC_PER_NODE=2 bash examples/python_extension/perf/run.sh
NPROC_PER_NODE=4 bash examples/python_extension/perf/run.sh
NPROC_PER_NODE=8 bash examples/python_extension/perf/run.sh
```

默认配置为 8 MiB、100 次迭代、10 次预热和 7 轮测量，可通过环境变量覆盖：

```bash
PERF_BYTES=8388608 \
PERF_ITERATIONS=100 \
PERF_WARMUP=10 \
PERF_ROUNDS=7 \
PERF_THRESHOLD_PERCENT=5 \
NPROC_PER_NODE=4 \
bash examples/python_extension/perf/run.sh
```

JSON 结果默认写入 ``examples/python_extension/perf/results/``。只有实际具备对应数量
NPU 的环境才能作为相应 2/4/8 卡验收证据，不得使用较少卡数的结果替代。

## 4. 跨机 ROCE handle_wait

``test/core/test_handle_wait.py`` 与 C++ ``examples/rdma_handlewait_test/use_handlewait`` 使用相同完成顺序：

```text
stream RMA -> handle_wait -> stream synchronize -> 数据可见性检查
```

在两台已配置 ROCE 的机器上，以标准多机 ``torchrun`` 参数启动同一脚本。例如每节点一张 NPU：

```bash
SHMEM_TEST_ENGINE=ROCE torchrun \
  --nnodes=2 \
  --nproc-per-node=1 \
  --node-rank=${NODE_RANK} \
  --master-addr=${MASTER_ADDR} \
  --master-port=${MASTER_PORT} \
  examples/python_extension/test/core/test_handle_wait.py
```

``NODE_RANK`` 在两台机器上分别为 0 和 1。``MASTER_ADDR`` 必须是两台机器可达的 rank 0 地址；CANN/驱动、ROCE 网卡和 SHMEM 路由需提前配置完成。

## 5. 用例与覆盖接口

|用例|覆盖内容|
|-|-|
|``test/core/test_memory.py``|``aclshmemx_malloc/calloc/align/free``、``MemType``、peer buffer|
|``test/core/test_sync_config_prof.py``|4 个 config 接口、4 个 Host 集合同步接口、2 个 stream barrier、``get_prof/show_prof``|
|``test/core/test_multi_instance.py``|UID 属性构造链路、``instance_ctx_get/set``、heap 隔离、指定实例 finalize|
|``test/core/test_handle_wait.py``|``Handle``、stream RMA、``handle_wait``、完成后数据可见性|
|原有 ``test/core`` 用例|初始化、direct、RMA/Signal 及兼容性回归|

``run.sh`` 默认给每个 ``torchrun`` 用例设置 10 分钟超时，可通过 ``SHMEM_TEST_TIMEOUT`` 调整。例如：

```bash
SHMEM_TEST_TIMEOUT=20m NPROC_PER_NODE=2 bash examples/python_extension/run.sh
```

超时通常意味着 PE 调用顺序不一致、某个进程提前异常退出，或运行环境的通信链路未就绪。
