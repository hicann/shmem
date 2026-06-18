## 环境要求

- 运行本示例需要机器具备RDMA环境（RDMA网卡及驱动已正确安装配置）。

### 检查RDMA环境
#### Ascend910B/C平台
```bash
for i in {0..7}; do hccn_tool -i $i -ip -g; done
for i in {0..7}; do hccn_tool -i $i -net_health -g; done
```
注意：7需要根据实际要查看的卡数修改。

可用环境命令输出如下：  
![](../../docs/images/rdma_env.png)

#### Ascend950平台
使用`ibv_devinfo`命令检查RDMA设备信息。
```bash
ibv_devinfo | grep xscale
```
可用环境命令输出如下：  
![](../../docs/images/nda-check.png)
## 使用方式
### 编译
在shmem/目录执行以下命令进行编译：
- Ascend910B/C 平台:
```bash
bash scripts/build.sh -enable_rdma -examples
```
- Ascend950 平台:
```bash
bash scripts/build.sh -soc_type Ascend950 -enable_rdma -rdma_backend XSCALE -examples
```
### 运行
#### 方式一：在`examples/rdma_demo`目录下执行`bash run.sh`

- 使用`run.sh`脚本执行

    `run.sh`支持通过`-pes`参数指定启动的PE数量，默认为2。
    ```bash
    bash run.sh -pes 4
    ```
    > 注：Ascend950平台需要在`run.sh`中设置`IBV_EXTEND_DRIVERS`环境变量：
    > ```bash
    > export IBV_EXTEND_DRIVERS=<path_to_libxscale_nda.so>
    > ```

#### 方式二：在shmem/目录手动运行命令
- 单机2卡执行命令
    ```bash
    export PROJECT_ROOT=<shmem-root-directory>
    export IBV_EXTEND_DRIVERS=<path_to_libxscale_nda.so> # 仅Ascend950平台需要
    export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
    ./build/bin/rdma_demo 2 0 tcp://127.0.0.1:8765 2 0 0 & # PE 0
    ./build/bin/rdma_demo 2 1 tcp://127.0.0.1:8765 2 0 0 & # PE 1
    ```
    > 注：\<shmem-root-directory\>为SHMEM项目的根目录。
- 跨机2卡执行命令

    假设机器A的ip为ip1，机器B的ip为ip2。
    在机器A执行如下命令：
    ```bash
    export PROJECT_ROOT=<shmem-root-directory>
    export IBV_EXTEND_DRIVERS=<path_to_libxscale_nda.so> # 仅Ascend950平台需要
    export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
    ./build/bin/rdma_demo 2 0 tcp://ip1:8765 1 0 0 # PE 0
    ```
    同时，在机器B执行如下命令：
    ```bash
    export PROJECT_ROOT=<shmem-root-directory>
    export IBV_EXTEND_DRIVERS=<path_to_libxscale_nda.so> # 仅Ascend950平台需要
    export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
    ./build/bin/rdma_demo 2 1 tcp://ip1:8765 1 1 0 # PE 1
    ```
    > 注：\<shmem-root-directory\>为SHMEM项目的根目录。
    >
    > 如需在容器中运行跨机测试，启动容器时指定`--net=host`模式即可。

3.命令行参数说明
```bash
    ./rdma_demo <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu>
```
- n_pes: 全局PE数量。
- pe_id: 当前进程的PE号。
- ipport: SHMEM初始化需要的IP及端口号，格式为tcp://<IP>:<端口号>。如果执行跨机测试，需要将IP设为PE0所在Host的IP。
- g_npus: 当前机器上启动的NPU卡的数量。
- f_pe: 当前机器上使用的第一个PE号。
- f_npu: 当前机器执行本样例使用的第一张NPU卡的卡号