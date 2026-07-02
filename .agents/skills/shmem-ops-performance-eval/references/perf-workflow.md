# 性能采集与对比（命令参考）

> **适用范围**：下列 perf 封装位于 **`custom-ops/scripts/`（skill 生成交付物）**，不是 SHMEM upstream。规范以本节为准；磁盘文件由 Phase 2/6 生成或同步。  
> **Skill 形态**：Markdown 代码段；**禁止**在 `.agents/skills/` 下放置 skill 附属 `.sh`/`.py`。  
> **SHMEM 原生**：仅 §0 环境链（`install/set_env.sh` 等）来自 upstream。  
> **Hard Gate**：HCCL baseline 与 SHMEM perf **NEVER** 同 shell / 同 `docker exec` 混跑。

环境见 [custom-ops-entrypoints.md §0](../../shmem-ops-compile-debug/references/custom-ops-entrypoints.md)。

---

## 1. 分阶段采集（推荐）

### 阶段 A — 仅 HCCL baseline

```bash
OP=<op_name>
DEVICE_LIST=0,1,2,3,4,5,6,7
PATTERN=uniform
BASE_COUNT=8388608
DTYPE=float16
ITERS=30

mkdir -p "${SHMEM_REPO}/custom-ops/${OP}/data/perf"
LOG="${SHMEM_REPO}/custom-ops/${OP}/data/perf/baseline_${PATTERN}_${BASE_COUNT}_${DTYPE}.log"

bash "${SHMEM_REPO}/custom-ops/${OP}/baseline/scripts/run_baseline.sh" \
  "${DEVICE_LIST}" "${PATTERN}" "${BASE_COUNT}" "${DTYPE}" "${ITERS}" \
  2>&1 | tee "${LOG}"
```

### 阶段 B — 仅 SHMEM

间隔 ≥30s，**新开 shell**：

```bash
OP=<op_name>
DEVICE_LIST=0,1,2,3,4,5,6,7
BASE_COUNT=8388608
DTYPE=float16
ITERS=30

mkdir -p "${SHMEM_REPO}/custom-ops/${OP}/data/perf"
LOG="${SHMEM_REPO}/custom-ops/${OP}/data/perf/shmem_uniform_${BASE_COUNT}_${DTYPE}.log"

bash "${SHMEM_REPO}/custom-ops/${OP}/scripts/perf.sh" \
  "${DEVICE_LIST}" "${BASE_COUNT}" "${DTYPE}" "${ITERS}" \
  2>&1 | tee "${LOG}"
```

### 阶段 C — 离线对比

```bash
OP=<op_name>
BASE_COUNT=8388608
DTYPE=float16

SHMEM_LOG="${SHMEM_REPO}/custom-ops/${OP}/data/perf/shmem_uniform_${BASE_COUNT}_${DTYPE}.log"
BASE_LOG="${SHMEM_REPO}/custom-ops/${OP}/data/perf/baseline_uniform_${BASE_COUNT}_${DTYPE}.log"

python3 "${SHMEM_REPO}/custom-ops/scripts/lib/perf_compare.py" \
  --shmem-log "${SHMEM_LOG}" --baseline-log "${BASE_LOG}"
```

时延算子可用 `custom-ops/scripts/perf_compare_latency.py` 对比 `kernel_us` / `comm_only_us`。

---

## 2. 平台区 sweep

**MUST** 在 per-PE payload ≥ 64MB 且相邻点带宽变化 ≤5% 的平台区取 `steady_bus_bandwidth_GBps`。

倍增 `base_count`：对每个 BASE 重复 §1 三阶段（A→B→C），**NEVER** 同 shell 混跑 HCCL 与 SHMEM。

Phase 6.5 优化轮次可单点采集；round 结束再 sweep 确认平台区。

---

## 3. 四算子默认参数

见 [platform-perf-spec.md](platform-perf-spec.md)。**8PE 不可用时**用 phase0 确认的 PE 数，并在报告中标注差异。

---

## 4. 产物布局

```
custom-ops/
├── scripts/lib/perf_compare.py
├── scripts/perf_compare_latency.py
└── <op_name>/
    ├── baseline/scripts/run_baseline.sh
    ├── scripts/perf.sh
    └── data/perf/
```

---

## 5. 达标口径

带宽：`SHMEM_steady_bus / HCCL_steady_bus`；时延：`HCCL / SHMEM`（comm_only 或 kernel_us）。  
**最终目标 MUST 以 [platform-perf-spec.md](platform-perf-spec.md) 按算子为准**（非 perf_compare.py 内置 105% 默认值）。

---

## 6. Docker

每阶段独立 `docker exec`（见 [docker-exec-contract.md](../../shmem-ops-dev/references/docker-exec-contract.md)）。
