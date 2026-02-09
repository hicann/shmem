### 使用方式

1. **编译项目**  
   在 `shmem/` 根目录下执行编译脚本：
   ```bash
   bash scripts/build.sh -examples
   ```

2. **运行KV_Shuffle示例程序**  
   进入示例目录并执行运行脚本：
   ```bash
   cd examples/kv_shuffle
   bash scripts/run.sh [pe_size]
   ```

   - **参数说明**：
     - `pe_size`：指定算子运行的pe个数。
     - 示例：使用第0和第1个NPU设备运行2卡kv_shuffle示例：
       ```bash
       bash scripts/run.sh 2
       ```
### 算子介绍
KV Shuffle 算子核心功能是实现 KV Cache（键值缓存）的跨设备 / 跨 PE 数据重排与远程拷贝，适配大模型训练 / 推理中 KV Cache 的分布式调度需求。

在大模型分布式训练 / 推理场景中，KV Cache 会按 Block（块）管理，不同计算 PE 之间需要根据调度策略（如 shuffle table）对 KV Cache 的 Block 进行重排、迁移，本算子为该场景提供高效 KV Block 跨 PE 拷贝与重映射，相比传统主机侧调度，大幅降低 KV Cache 迁移的延迟和带宽开销。

#### 执行流程

本算子采用主从 PE 配对执行模式（一个 PE 作为发送方，配对 PE 作为接收方），整体流程如下：
* 算子初始化阶段
算子实例化时完成共享内存分配和初始化，为跨 PE 执行做准备：
根据总 PE 数、核数、同步标志间隔，通过aclshmem_malloc分配跨 PE 可见的同步标志内存（sync_ptr_）；
通过aclrtMemset将同步标志内存初始化为 0，保证初始同步状态干净。
* 核函数启动与硬件初始化
获取分布式基础参数：总 PE 数（n_pes）、本地 PE（local_PE）；
从global_shuffle_table解析配对 PE（pair_PE） 和本地操作类型（operation）、配对 PE 操作类型（pair_operation）（0 为发送，非 0 为接收）；
计算核内基础参数：总核数（aiv_num）、当前核索引（aiv_idx），以及本地 / 配对 PE 的同步标志偏移量（保证每个核有独立的同步位，避免核间冲突）；
划分 UB 乒乓缓存：将 32KB UB 分为 K 的 Ping/Pong 缓存（0~32KB、32~64KB）和 V 的 Ping/Pong 缓存（64~96KB、96~128KB），缓存大小均为 32KB。
* 跨 PE 执行角色判定与前置同步
根据global_shuffle_table中的operation判定当前 PE / 核的执行角色（发送方 / 接收方），并完成前置时序同步：
角色 1：发送方（operation == 0）
等待接收方就绪：通过aclshmem_signal_wait_until监听配对 PE 的同步标志位，直到接收方完成初始化并将标志位设为count，保证 “先准备后发送”；
角色 2：接收方（pair_operation == 0，即配对 PE 为发送方）
设置本地就绪标志：通过aclshmemx_signal_op将本地同步标志位设为count，通知发送方 “接收方已就绪”；
等待发送方完成：监听发送方的同步标志位，直到发送方完成所有 KV Block 拷贝，保证接收方不提前退出。
* 跨 PE 拷贝
步骤 1：获取 Block 重排映射关系
从src_block_table/dst_block_table中读取当前 Block 的源 ID和目标 ID。
步骤 2：计算数据偏移地址
根据 Block ID、核索引计算当前核需要处理的 KV 数据在全局内存中的源偏移和目标偏移：
源偏移：src_offset = src_block_id * block_size + (aiv_idx % 8) * core_size；
目标偏移：dst_offset = dst_block_id * block_size + (aiv_idx % 8) * core_size；
注：aiv_idx % 8保证核间分工的偏移一致性，前 8 核处理 K，后 8 核处理 V，互不干扰。
步骤 3：乒乓缓存与事件初始化
切换乒乓缓存标志位（ping_pong_flag），为本次拷贝分配 UB 缓存和硬件事件：
K/V 分别使用独立的 Ping/Pong UB 缓存，避免数据覆盖；
为每个拷贝操作分配独立的 MTE 事件（EVENT_ID0~3），用于拷贝完成等待。
步骤 4：核间分工执行 MTE 非阻塞远程拷贝
基于核索引实现K/V 核间分工（前 8 核处理 K Cache，后 8 核处理 V Cache），通过 SHMEM 非阻塞拷贝接口（aclshmemx_mte_put_nbi） 完成跨 Rank 远程拷贝：
K Cache 拷贝（aiv_idx < 8）：
取 K Cache 的源 / 目标全局内存地址；
调用aclshmemx_mte_put_nbi：以当前核 UB 为临时缓存，将本地 K Cache 数据通过 MTE 引擎拷贝到配对 Rank 的目标 K Cache 地址；
通过SetFlag/WaitFlag操作 MTE 硬件事件，等待拷贝完成，保证数据原子性（非阻塞接口需显式等待，避免数据未拷贝完成就切换缓存）。
V Cache 拷贝（aiv_idx >= 8）：
逻辑与 K Cache 一致，使用独立的 UB 缓存和 MTE 事件，避免与 K Cache 的资源冲突。
步骤 5：切换乒乓缓存标志位
完成当前 Block 拷贝后，将ping_pong_flag取反（0→1/1→0），为下一个 Block 的拷贝分配另一组 UB 缓存和事件，实现缓存流水线，隐藏数据拷贝的延迟。
* 执行完成同步与资源释放
发送方完成所有 Block 拷贝后，通过aclshmemx_signal_op设置本地同步标志位为count，通知接收方 “发送完成”；
接收方监听到发送方的完成标志后，核函数执行结束；
算子销毁时（~KVShuffleOps），通过aclshmem_free释放跨 PE 同步内存，避免内存泄漏；
算子compute接口中，每次执行后count_自增，保证多轮执行的同步标志不重复，避免跨轮冲突。