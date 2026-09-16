# HCCS SIO Link

The HCCS/SIO link test tool is designed to verify the correctness of SIO and HCCS links between NPUs.

> **Dependencies**:
>
> - CANN version >= 9.1.0
> - Ascend HDK version >= 26.0.0
> - LingQu Computing Network version >= 1.5.3. Download address: [LingQu Computing Network](https://support.huawei.com/enterprise/zh/ascend-computing/lingqu-computing-network-pid-258003841/software)

## Link Description

In the A3 chip, each NPU contains two dies. The dies are interconnected via the original SIO link. Building upon this existing SIO link, the solution introduces an additional HCCS link, enabling dual-path parallel transmission via both SIO and HCCS. This architecture significantly accelerates inter-die data transfer.

- **SIO**: the original interconnect link between dies. After SHMEM initialization, the default virtual address (VA) access to the peer die is routed through the SIO link.

- **HCCS**: a new interconnect link between dies. By calling [`aclrtMemMapSelectedLink`](https://gitcode.com/cann/runtime/blob/master/docs/zh/api_ref/11-04_virtual_memory_management.md#aclrtmemmapselectedlink), a new VA can be mapped to the same physical address (PA) as the original SIO VA. `ACL_RT_MEM_LINK_IDX_1` is used to select an HCCS link. In this case, access through the new VA is routed via the HCCS link.

- **SIO + HCCS dual-link parallel transmission**: Data can be transmission simultaneously over both SIO and HCCS links, fully utilizing dual-link bandwidth to enhance transmission performance.

## Core Functions

### `setup_hccs_mapping`

An HCCS link mapping is set up to create a VA channel for the current PE for accessing the heap memory of the peer PE through the HCCS link. This function involves the following steps:

1. **Obtaining a local heap base address**: Call `aclshmemx_get_heap_base()` to obtain the symmetric heap base address of the current PE.
2. **Translating to a peer address**: Call `aclshmem_ptr()` to convert the local heap base address into a peer-accessible VA (`peer_heap_base`). This address is routed through the SIO link.
3. **Reserving a VA space**: Call `aclrtReserveMemAddress()` to reserve an unmapped VA range (`hccs_ptr`) in the VA space of the current PE.
4. **Mapping to an HCCS link**: Call [`aclrtMemMapSelectedLink`](https://gitcode.com/cann/runtime/blob/master/docs/zh/api_ref/11-04_virtual_memory_management.md#aclrtmemmapselectedlink) to map the reserved VA to the PA corresponding to `peer_heap_base` and specify `ACL_RT_MEM_LINK_IDX_1` to select an HCCS link. At this point, accesses via `hccs_ptr` are routed through the HCCS link, while accesses via the original SIO VA continue to use the SIO link. Both VAs point to the same PA but traverse different physical links.

> **Key principle**: The SIO and HCCS links share the same PA, but the traffic is distributed using different VAs and link indexes, thus supporting dual-link parallel transmission.

#### Parameter Description

| Parameter| Type| Description|
|------|------|------|
| `peer` | `int` | ID of the peer PE, that is, the target PE for which the HCCS mapping is to be established.|
| `local_mem_size` | `uint64_t` | Identical to the symmetric heap size (in bytes) specified during `aclshmemx_init_attr` initialization. The actual mapped range size is `local_mem_size + ACLSHMEM_EXTRA_SIZE`.|
| `hccs_ptr` | `void **` | Output parameter. Upon success, the function returns the base virtual address of the HCCS link mapping. If it fails, the value depends on the step at which the error occurred. If `aclrtReserveMemAddress` succeeds but subsequent steps fail, `*hccs_ptr` may contain a reserved but unmapped address. In this case, you need to call `teardown_hccs_mapping` or `HccsMappingGuard` to release the addresses to prevent leaks.|

#### Return Values

- `true`: The HCCS mapping is successfully established, and `*hccs_ptr` is a valid HCCS base virtual address.
- `false`: The mapping fails at one of the steps. In this case, the function prints an error message and returns.

### `teardown_hccs_mapping`

This function is paired with `setup_hccs_mapping` and is used to release HCCS mapping resources:

1. Call `aclrtUnmapMem()` to remove the mapping between the VA and PA.
2. Call `aclrtReleaseMemAddress()` to release the reserved VA space.

> **Note**: In the sample code, the `HccsMappingGuard` structure automatically calls `teardown_hccs_mapping` during destruction using the RAII mechanism to ensure that mapping resources are not leaked.

## Build

To build this example, enable the `-examples` build option. During the build process, CMake automatically detects whether the current CANN version supports the `aclrtMemMapSelectedLink` function. If it does, CMake automatically builds this example:

> **Atlas A3 only**: This example depends on the inter-die SIO/HCCS links and is supported only on A3 (Atlas A3 training series/Atlas A3 inference series). It is not supported on A2 (Atlas A2 training series/Atlas A2 inference series) or Ascend950.

```bash
bash scripts/build.sh -examples
```

Build output: `build/bin/hccs_sio_link`

## Prerequisites

- The SHMEM project has been built as described above.
- The environment variable `ASCEND_HOME_PATH` has been properly configured.

> **Multi-instance description**: The function `aclshmemx_get_heap_base` returns the heap base address of the currently active instance. In multi-instance scenarios, you must first switch to the target instance using `aclshmemx_instance_ctx_set_impl`, and then call `aclshmemx_get_heap_base`.

## Usage

This tool is started using the `run.sh` script. The script starts a background process for each PE, and communication between PEs is established through SHMEM initialization.

> **Running the binary directly**: In performance test modes, `run.sh` creates the `output` directory automatically. If you bypass the script and run `build/bin/hccs_sio_link` directly, first run `mkdir -p output` in the process working directory; otherwise, the performance CSV files cannot be written.

### Run Command

```bash
bash run.sh [option]
```

### Typical Cases

```bash
# Default configuration: two PEs, full-link test for SIO + HCCS, 4 KB of data, and int type
bash run.sh

# If four PEs are specified, test the HCCS link only.
bash run.sh -pes 4 -mode hccs

# If eight PEs, with 8 KB of data and the fp32 type, are specified, test the SIO link only.
bash run.sh -pes 8 -size 8 -type fp32 -mode sio

# SIO + HCCS hybrid test (3/5 of data via SIO and 2/5 via HCCS)
bash run.sh -mode mixed

# Hybrid Get performance test (PE 0 collects cycle data)
export SHMEM_CYCLE_PROF_PE=0
bash run.sh -mode mixed_get_perf

# Hybrid Put performance test (PE 1 collects cycle data)
export SHMEM_CYCLE_PROF_PE=1
bash run.sh -mode mixed_put_perf
```

## Parameter Description

| Parameter| Default Value| Description|
|------|--------|------|
| `-ipport` | `tcp://127.0.0.1:8766` | Communication initialization address|
| `-pes` | `2` | Total number of PEs (same as the number of NPUs) involved in the test|
| `-fpe` | `0` | ID of the first PE|
| `-fnpu` | `0` | ID of the first NPU|
| `-type` | `int` | Test data type: `int` / `int64` / `fp32`|
| `-mode` | `all` | Test mode (see the table below)|
| `-size` | `4` | Data size (KB) of each PE|

### Test Modes

| Mode| Description|
|------|------|
| `sio` | SIO link correctness test|
| `hccs` | HCCS link correctness test|
| `all` | SIO + HCCS full-link correctness test|
| `mixed` | SIO + HCCS hybrid correctness test (3/5 data via SIO and 2/5 via HCCS)|
| `mixed_get_perf` | SIO + HCCS hybrid Get performance test (3/5 data via SIO and 2/5 via HCCS)|
| `mixed_put_perf` | SIO + HCCS hybrid Put performance test (3/5 data via SIO and 2/5 via HCCS)|

## Performance Test

The `mixed_get_perf` and `mixed_put_perf` modes are used to measure the performance (in cycles) of SIO + HCCS dual-link parallel transmission. They test the Get (remote read) and Put (remote write) operations respectively.

### Working Principle

- Data is distributed to the SIO and HCCS links at a ratio of 3:2 (3/5 of data via SIO and 2/5 via HCCS).
- Inside the kernel, data is partitioned by block: some blocks handle SIO link transmission, and the other blocks handle HCCS link transmission.
- `aclshmemx_mte_get_nbi` / `aclshmemx_mte_put_nbi` are used for non-blocking DMA transmission.
- A cycle counter collects the duration of each transmission, and the shmem profiling mechanism (`aclshmemx_get_prof`) is used to output statistics.

### Environment Variables

| Environment Variable| Description|
|----------|------|
| `SHMEM_CYCLE_PROF_PE` | ID of the PE that performs performance collection. The default value is `0`. Only this PE executes the cycle collection logic, and the other PEs only participate in barrier synchronization.|

### Run Examples

```bash
# Hybrid Get performance test: two PEs, 4 KB of data, int type, and PE 0 collects performance data
export SHMEM_CYCLE_PROF_PE=0
bash run.sh -mode mixed_get_perf

# Hybrid Put performance test: two PEs, 8 KB of data, fp32 type, and PE 1 collects performance data
export SHMEM_CYCLE_PROF_PE=1
bash run.sh -pes 2 -size 8 -type fp32 -mode mixed_put_perf
```

### Internal Parameters

The internal parameters of the performance test are defined in `utils/hccs_sio_link_config.h`:

| Parameter| Constant| Value| Description|
|------|--------|----|------|
| Number of kernel blocks | `HCCS_SIO_BLOCK_DIM` | `32` | Total number of blocks launched by the kernel|
| UB buffer size | `HCCS_SIO_UB_SIZE_KB` | `16` | Unified Buffer size (KB) of each block|
| SIO ratio numerator | `HCCS_SIO_RATIO_NUM` | `3` | SIO data volume = total × NUM / DEN|
| SIO ratio denominator | `HCCS_SIO_RATIO_DEN` | `5` | HCCS data volume = total - SIO data volume|
| Minimum data size | `HCCS_SIO_PERF_MIN_LOG2_BYTES` | `4` | log2(bytes) of the minimum transfer data volume, 4 = 16 B|
| Maximum data size | `HCCS_SIO_PERF_MAX_LOG2_BYTES` | `20` | log2(bytes) of the maximum transfer data volume, 20 = 1 MB|
| Data size step | `HCCS_SIO_PERF_STEP_LOG2` | `1` | log2 step, 1 = doubling each time|
| Warmup rounds | `HCCS_SIO_PERF_WARMUP` | `100` | Number of warmup iterations (not included in statistics)|
| Test rounds | `HCCS_SIO_PERF_LOOP_COUNT` | `1000` | Number of measured iterations|
| Unidirectional/bidirectional mode | `HCCS_SIO_PERF_IS_UNILATERAL` | `true` | `true` = unidirectional (only prof_pe executes); `false` = bidirectional (all PEs execute)|

> **Note**: Performance collection is limited by `ACLSHMEM_CYCLE_PROF_MAX_BLOCK` (maximum number of recorded cores) and `ACLSHMEM_CYCLE_PROF_FRAME_CNT` (maximum number of recorded frames). Blocks or frames beyond these limits are not recorded.

## Output Example

Correctness test:

```text
PE 0: [SIO] path verification PASSED for PE 1
PE 1: [SIO] path verification PASSED for PE 0
PE 0: [HCCS] path verification PASSED for PE 1
PE 1: [HCCS] path verification PASSED for PE 0
```

Hybrid test:

```text
PE 0: [MIXED-SIO] path verification PASSED for PE 1
PE 0: [MIXED-HCCS] path verification PASSED for PE 1
PE 1: [MIXED-SIO] path verification PASSED for PE 0
PE 1: [MIXED-HCCS] path verification PASSED for PE 0
```
