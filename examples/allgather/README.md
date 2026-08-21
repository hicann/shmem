# 样例介绍

本样例演示基于 SHMEM 的 allgather 集合通信：各 PE 将本地数据汇聚到所有 PE，同时完成精度验证，性能数据输出在 `result.csv` 中。

使用方式:

1. 在shmem/目录编译:

    - A2/A3 平台:

    ```bash
    bash scripts/build.sh -examples
    ```

    - Ascend950 平台:

    ```bash
    bash scripts/build.sh -soc_type Ascend950 -examples
    ```

2. 在shmem/examples/allgather目录执行demo:

    ```bash
    bash run.sh -pes ${PEs} -type ${TYPEs}
    ```

## 参数说明

`run.sh` 支持以下参数：

| 参数 | 默认值 | 说明 |
| ---- | ---- | ---- |
| `-pes` | 2 | 参与通信的 PE 数量，可选值：2、4、8 |
| `-type` | int32_t | 传输的数据类型，可选值：int32_t、float16_t、bfloat16_t |
| `-ipport` | tcp://127.0.0.1:8766 | 初始化监听的 IP 和端口，需保证端口空闲可绑定 |
| `-gnpus` | 8 | 使用的 NPU 卡数。若大于 `-pes` 指定的 PE 数，脚本会自动将其调整为 PE 数 |
| `-fnpu` | 0 | 起始 NPU 卡号，多卡按卡号连续递增分配 |
| `-fpe` | 0 | 起始 PE 号 |
| `-tool` | msprof | 性能采集工具，当前仅支持 msprof |

## 运行示例

```bash
# 2个PE，数据类型int32_t
bash run.sh -pes 2 -type int32_t

# 4个PE，数据类型float16_t，从NPU卡号2开始使用4张卡
bash run.sh -pes 4 -type float16_t -fnpu 2
```

运行结束后，精度验证结果输出到控制台，性能数据输出在 `result.csv` 中。
