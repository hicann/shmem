# 经典 MoE Dispatch 示例

本示例实现非量化 MoE 的经典 dispatch 算子，对应设计文档 `DOC/moe_dispatch_combine_non_quant_architecture.md`。Dispatch 的职责是根据 `expert_ids` 将本 PE 的 token/topK 路由到目标 expert 所在 PE，并生成 combine 阶段需要的辅助信息。

## 功能说明

输入包括：

- `x`：本 PE 的 token hidden states。
- `expert_ids`：每个 token/topK 对应的全局 expert id。

输出包括：

- `expand_x`：按目标 expert 聚合后的 token 数据。
- `assist_info_for_combine`：combine 回传使用的辅助信息，格式为 `[src_rank_id, src_token_id, src_topk_id]`。
- `ep_recv_count`：按 `(local_expert_id, src_rank_id)` 排列的累计接收计数。
- `expert_token_nums`：每个本地 expert 实际收到的 token 数。

输出顺序固定为 `(local_expert_id, src_rank_id)`，便于后续 combine 和 golden 校验。

## 实现逻辑

Kernel 启动 `pe_size` 个 AIV core。发送阶段中，`core_id == dst_rank` 的 core 负责写入对应目标 PE 的对称窗口。

主要流程：

1. 每个 active core 扫描本 PE 的 `expert_ids`，统计发往自己负责目标 PE 的每个 local expert 的 token 数。
2. 对每个命中的 token/topK，通过 MTE `put_nbi` 写 payload 到目标 PE 的 SHMEM window。
3. 写入 `assist_info_for_combine`、token ready flag 和 segment count 信号。
4. 所有发送 core 完成后，接收端等待各来源 rank 的 count 信号。
5. 根据 count 构造 `ep_recv_count` 和 `expert_token_nums`，再按确定顺序 compact 到最终输出。

当前经典版本优先保证逻辑清晰、可验证和输出顺序稳定，数据面使用 MTE 传输。

## 构建

在仓库根目录执行：

- A2/A3 平台:

```bash
bash scripts/build.sh -examples
```

- Ascend950 平台:

```bash
bash scripts/build.sh -soc_type Ascend950 -examples
```

## 运行

基础 2 卡测试：

```bash
cd examples/dispatch/dispatch_classic
bash scripts/run.sh -pes 2 -bs 8 -h 16 -topk 2 -expertPerPe 2 -type int32_t
```

8 卡、64 expert 测试：

```bash
cd examples/dispatch/dispatch_classic
bash scripts/run.sh -pes 8 -bs 8 -h 16 -topk 2 -expertPerPe 8 -type int32_t
```

脚本会自动生成输入、路由矩阵和 golden 数据，启动每个 PE 对应的进程，输出写入 `output/`，并校验 `expand_x`、`assist_info`、`ep_recv_count` 和 `expert_token_nums`。

## 参数说明

```text
-pes <n>              PE 数量，单机示例要求与 -gnpus 相同（默认 2）。
-gnpus <n>            本机启动的 NPU 数量，必须与 -pes 相同（默认 2）。
-bs <n>               每个 PE 的 token 数（默认 8）。
-h <n>                token hidden size（默认 16）。
-topk <n>             每个 token 路由的 expert 数（默认 2）。
-expertPerPe <n>      每个 PE 上的 local expert 数，范围为 [1, 1024]（默认 2）。
-type <dtype>         数据类型，支持 int32_t、float16_t、bfloat16_t（默认 int32_t）。
-fpe <id>             首个 PE 编号（默认 0）。
-fnpu <id>            起始 NPU id（默认 0）。
-ipport <url>         SHMEM bootstrap 地址（默认 tcp://127.0.0.1:8766）。
--perf                性能测试模式：保留正确性校验，并将性能 CSV 写入 --output-dir 目录。
--warmup <n>          性能测试的预热迭代次数，不计入统计（默认 5）。
--loops <n>           性能测试的正式测量迭代次数（默认 50）。
-pes-list <a,b,...>   性能测试模式下扫描的 PE 数量列表。
-bs-list <a,b,...>    性能测试模式下扫描的每个 PE token 数列表。
-h-list <a,b,...>     性能测试模式下扫描的 hidden size 列表。
--topk-list <a,b,...> 性能测试模式下扫描的 topk 列表。
--expert-per-pe-list <a,b,...>
                      性能测试模式下扫描的 local expert 数列表。
--prof-pe <id|all>    性能采集的 PE 编号；为 all 时轮流采集每个 PE 并汇总（默认 0）。
--output-dir <dir>    性能测试 CSV 输出目录（默认 output/perf）。
-a|--analyse <mode>   性能结果处理方式：plot（图形化展示）、md（生成 Markdown 报告）、none（不处理，默认）。
```

`expertPerPe` 上限为 1024。Kernel 在 AI core 栈上为每个目标 local expert 分配固定工作区，超过该上限会被 host 侧拒绝。

## 性能测试

`scripts/run.sh --perf` 会在每个 shape 上保留正确性校验，并在 `output/perf/` 下写入 CSV。

单 shape profiling：

```bash
cd examples/dispatch/dispatch_classic
bash scripts/run.sh --perf -pes 2 -bs 8 -h 256 -topk 2 -expertPerPe 2 -type int32_t \
    --warmup 5 --loops 50
```

多 shape、多卡数 sweep：

```bash
cd examples/dispatch/dispatch_classic
bash scripts/run.sh --perf --pes-list 2,4,8 --bs-list 8,16,32 --h-list 64,256,1024 \
    --topk-list 2 --expert-per-pe-list 2,8 -type int32_t --prof-pe all \
    --warmup 5 --loops 50
```

CSV 指标包括：

- `full_op`：完整 dispatch，包括通信、元数据构造、compact 和同步。
- `comm_only`：Stage 1 payload 通信及必要的元数据/status 协议。

单 rank 文件名为 `dispatch_perf_rank<rank>.csv`。使用 `--prof-pe all` 时，脚本会轮流 profile 每个 PE，并生成 `dispatch_perf_summary.csv`。
