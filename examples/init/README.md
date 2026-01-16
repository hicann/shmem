### 使用方式

#### 1. 编译项目

在 `shmem/` 根目录下执行编译脚本：

默认后端
```bash
bash scripts/build.sh
source install/set_env.sh
```

mf后端
```bash
bash scripts/build.sh -mf
source install/set_env.sh
```

mf后端仅支持default编译

##### 2. 运行 init 用例

用例目录
```bash
cd examples/init
```

用例接收两个参数，第一个参数指定流程，第二个参数指定pe数量（不能超过卡数）。不指定时默认`mode = default`,`pesize = 2`。
执行 default 流程，2 pe
```bash
bash run.sh -mode default -pesize 2
```

执行 mpi 流程，2 pe （依赖MPI，不支持mf后端）
```bash
bash run.sh -mode mpi -pesize 2
```

执行 uid 流程，2 pe（依赖MPI，不支持mf后端）
```bash
bash run.sh -mode uid -pesize 2
```

2，3流程执行前需要自行安装并导入MPI环境变量

```bash
# mpich安装在默认路径时可参考，需根据实际情况替换。
export PATH=/usr/local/mpich/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/mpich/lib:$LD_LIBRARY_PATH
```