# 性能相关 Skill 审视备忘

> 供 Agent 维护 skill 时参考；交付给用户时引用关键结论即可。

## 已覆盖能力（2026-06 更新）

| 能力 | 文档位置 |
| --- | --- |
| baseline C++ 接入 + perf_compare | baseline-compare-workflow.md |
| 稳定带宽 sweep + steady_bus | baseline-compare-workflow.md §1.1 |
| Phase 6 聊天自动输出 | perf-chat-output-spec.md §2 |
| Phase 6.5 每轮 Δ% 自动输出 | perf-chat-output-spec.md §3~§4 |
| 报告模板 steady_bus 列 | templates/performance-report.md |

## 仍待工程化（非 skill 条文能单独解决）

1. **连续 8PE 测试资源冲突** → 已改为 **HCCL/SHMEM 分阶段 + 离线对比**（见 [perf-workflow.md](perf-workflow.md)）
2. ~~HCCL 不得在 SHMEM init 之后调用~~ → skill + 脚本已强制隔离
3. **sweep 耗时长**：skill 允许优化轮次内 step 仅用算子 `scripts/perf.sh`，round 最优再跑 perf-workflow §2
4. **peak_bandwidth 单链路 28GB/s**：utilization>100% 需在报告中注释；未改公式
5. **Device Frame 未强制自动化**：eval 仍要求 §3.4/§4 Frame 表，采集依赖 SHMEMI_PROF 手工步骤

## 历史不一致项（已修正）

| 项 | 原状 | 现规范 |
| --- | --- | --- |
| 达标指标 | optim 用 e2e_latency，eval 用 steady_bus | 四算子见 [platform-perf-spec.md](platform-perf-spec.md)；其他有 baseline 算子默认 steady_bus ≥ 80% baseline |
| Phase 6.5 入口 | dev 写「未达标才进入」与 optim「达标也跑 5 轮」矛盾 | **未达标 + `performance_auto_optim:true` 才进入**；进入后 **MUST 跑满 5 轮** |
| Round 1 性质 | 多处写「参数调优轮」 | **Round 1~5 均为机制优化轮**；参数扫描为同轮补充 |
| Phase 0 题数 | 四项 vs 五项混用 | **五项 AskQuestion**（#5 = `performance_auto_optim`） |
| Skill/仓路径 | skill 相对路径链到 `docs/` | **先定位 `SHMEM_REPO`**（shmem-repo-resolution.md） |
| 聊天输出 | 「不得等用户追问」分散多处 | 集中 **perf-chat-output-spec.md** |
| 带宽结论 | 单点 M 档 e2e 带宽 | **MUST sweep 找平台区** |

## 建议后续增强（可选）

- 在 `custom-ops/scripts/` 增加 sweep 薄封装（可选；规范见 perf-workflow.md §2）
- performance_report §5 轮次总览与聊天累计表字段对齐代码生成
- timing-and-metrics-standard.md 增补 steady_bus 公式与 sweep 门禁交叉引用
