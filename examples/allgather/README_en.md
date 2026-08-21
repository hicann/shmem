# Introduction

This sample demonstrates SHMEM-based allgather collective communication: each PE gathers local data to all PEs, with precision verification completed at the same time. Performance data is output in `result.csv`.

Instructions:

1. Build in the `shmem/` directory.
```
bash scripts/build.sh -examples
```
2. Run the demo in the `shmem/examples/allgather` directory:
```
bash run.sh -pes ${PEs} -type ${TYPEs}
```

## Parameters

`run.sh` supports the following parameters:

| Parameter | Default Value | Description |
| ---- | ---- | ---- |
| `-pes` | 2 | Number of PEs participating in communication. Valid values: 2, 4, and 8 |
| `-type` | int32_t | Data type to transfer. Valid values: int32_t, float16_t, and bfloat16_t |
| `-ipport` | tcp://127.0.0.1:8766 | IP address and port for initialization listening. Ensure that the port is idle and can be bound |
| `-gnpus` | 8 | Number of NPU cards to use. If this value is greater than the number of PEs specified by `-pes`, the script automatically adjusts it to the number of PEs |
| `-fnpu` | 0 | First NPU card ID. Multiple cards are allocated in ascending order of card IDs |
| `-fpe` | 0 | First PE ID |
| `-tool` | msprof | Performance collection tool. Currently, only msprof is supported |

## Running Examples

```bash
# Two PEs and the int32_t data type
bash run.sh -pes 2 -type int32_t

# Four PEs and the float16_t data type, using four cards starting from NPU card 2
bash run.sh -pes 4 -type float16_t -fnpu 2
```

After the running is complete, the precision verification result is output to the console, and the performance data is output in `result.csv`.
