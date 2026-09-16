# Sample Introduction

## Version and Platform Support

- The UDMA sample supports only the Ascend950 platform. Other platforms do not support this sample.
- CANN 9.1.0 provides the HCOMM resource APIs required by UDMA, such as `HcommEndpointCreate`, `HcommMemReg`, `HcommChannelCreate`, and `HcommChannelGetStatus`. Download and install the toolkit and ops packages corresponding to Ascend950 from the [CANN 9.1.0 resources](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.1.0). CANN versions earlier than 9.1.0 are not supported by this sample.
- During initialization, the required HCOMM symbols are loaded and checked. Before building, run `source /usr/local/Ascend/ascend-toolkit/set_env.sh` (use the corresponding `set_env.sh` for a custom installation path) and build with `-soc_type Ascend950`. If the HCOMM runtime library or symbols are incomplete, the initialization fails.
- After initialization, the sample enables `aclshmemx_enable_exception_report(nullptr, ACLSHMEMX_EXCEPTION_REPORT_DEBUG)` and calls `aclshmemx_report_exception()` after kernel synchronization to demonstrate how to call the UDMA exception reporting APIs.

Instructions:

1. Build in the `shmem/` directory:
```bash
bash scripts/build.sh -examples -soc_type Ascend950
```

2. Run in the `shmem/` directory:
```bash
bash examples/udma_demo/run.sh 0 # AllGather test
bash examples/udma_demo/run.sh 1 # put signal test
```
By default, the script launches with 8 devices on a single server, sequentially starting `PE 0` to `PE 7`, and waits for all processes to exit.

`run.sh` automatically sets `PROJECT_ROOT`, `LD_LIBRARY_PATH`, and `SHMEM_UID_SESSION_ID` (based on the `ipport` parameter). You do not need to manually export these environment variables.

The UDMA advanced RMA APIs use `PIPE_MTE3` to submit WQEs by default, which requires a UB scratch area. The default configuration is `offset = 189 * 1024`, `ub_size = 128` bytes, and `sync_id = 0`. If the sample or a service kernel needs to reuse this UB area, adjust it via `aclshmemx_set_udma_config(offset, ub_size, sync_id)`. `ub_size` must be at least 128 bytes to hold a complete WQE staging block for the current UDMA data movement operation.

3. Command-line parameters of the run.sh script

The script parses parameters by position. All parameters are optional, and default values are used if they are not specified.
```bash
bash examples/udma_demo/run.sh <test_type> <n_pes> <g_npus> <ipport> <f_pe> <local_pes> <f_npu>
```

- test_type: test type. Value 0 (default) indicates the all-gather test, and value 1 indicates the put signal test.
- n_pes: total number of global PEs across all servers (default 8).
- g_npus: number of NPU devices used on the local server (default 8).
- ipport: bootstrap node address, in the format `IP:PORT` (default `127.0.0.1:8899`). It must be the IP address of node0 and reachable from all nodes.
- f_pe: first global PE ID used on the local server (default 0).
- local_pes: number of PE processes to start on the local server (default: equal to n_pes).
- f_npu: ID of the first NPU device used on the local server (default 0).

Multi-server launch example (2 servers x 8 devices = 16 PEs; assume the IP address of node0 is 192.168.1.10):
```bash
# node0: f_pe=0, starting global PEs 0 to 7
bash examples/udma_demo/run.sh 0 16 8 192.168.1.10:8899 0 8 0
# node1: f_pe=8, starting global PEs 8 to 15
bash examples/udma_demo/run.sh 0 16 8 192.168.1.10:8899 8 8 0
```

### mssanitizer Check

Before running mssanitizer, rebuild with `-mssanitizer`:

```bash
bash scripts/build.sh -examples -soc_type Ascend950 -mssanitizer
```

Currently, the experiment restores the struct field padding on the MTE3 path, and both the all-gather and put signal kernels perform a one-time zeroing of the entire 128-byte scratch area before their first call (`init_udma_wqe_scratch` in `udma_demo_kernel.cpp`). This initializes the original values that may be read during the read-modify-write of the struct bit fields and also covers the 32-byte tail padding of put signal. It is used to verify whether full pre-initialization by the caller can eliminate uninitialized reads reported by initcheck.

```bash
mssanitizer -- bash examples/udma_demo/run.sh 0
mssanitizer -- bash examples/udma_demo/run.sh 1
```

4. Command-line parameters of the underlying binary
```bash
./udma_demo <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu> [test_type]
```

- n_pes: number of global PEs.
- pe_id: PE ID of the current process.
- ipport: IP address and port number required for SHMEM initialization, in the format tcp://`<IP_address>:<port_number>`.
- g_npus: number of NPUs started on the current server.
- f_pe: ID of the first PE used on the current server.
- f_npu: ID of the first NPU used to run this sample on the current server.
- test_type: test type (optional). Value 0 (default) indicates the all-gather test, and value 1 indicates the put signal test.
