# 编译与构建
## SHMEM编译
### 下载SHMEM源码

```shell
git clone https://gitcode.com/cann/shmem.git
```

您可自行选择需要的分支。

### 编译

进入shmem的根目录，编译

```shell
cd shmem
bash scripts/build.sh
```

更多命令介绍可查看shmem仓主目录下的`README.md`和`scripts/build.sh`文件。

### SHMEM编译相关说明
SHMEM的基本编译命令是`bash build.sh`，默认构建模式下生成版本信息，并创建安装包(默认情况下不会编译RDMA能力、用例、测试、python接口)。后可跟参数，实现不同功能：
- `-use_cxx11_abi1`：启用 `C++11 ABI`，默认
- `-use_cxx11_abi0`：禁用 `C++11 ABI`
- `-mf`：启用MF后端
- `-debug`：设置构建类型为 `Debug` 模式
- `-examples`：构建examples目录下所有用例
- `-enable_rdma`：构建并启用RDMA相关能力
- `-enable_ascendc_dump`：启用`AscendC_Dump`模式，用于对算子内核代码进行调测
- `-package`：构建py扩展的whl包
- `-gendoc`：生成文档
- `-onlygendoc`: 生成文档，不构建源码

### SHMEM关键文件介绍
1. `scripts`目录：
   - `install.sh`: 安装脚本
   - `uninstall.sh`: 卸载脚本
   - `build.sh`: 编译脚本
   - `release.sh`：全自动构建与打包脚本
   - `set_env.sh`：SHMEM的环境变量设置文件

## 配置文件

#### 编译文件`build.sh`

文件名：`scripts/build.sh`  
SHMEM编译文件，一般无需更改。

#### 环境变量设置文件`set_env.sh`

​**文件名**​：`scripts/set_env.sh`  
SHMEM安装完成后，提供进程级环境变量设置脚本`set_env.sh`，以自动完成环境变量设置，用户进程结束后自动失效。
