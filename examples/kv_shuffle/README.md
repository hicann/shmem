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