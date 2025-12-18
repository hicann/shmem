# 代码组织结构
## ACLSHMEM组织结构
``` 
├── 3rdparty // 依赖的第三方库
├── docs     // 文档
├── examples // 使用样例
├── include  // 头文件
├── scripts  // 相关脚本
├── src      // 源代码
└── tests    // 测试用例
```

## include
```
include目录下的头文件是按照如下文件层级进行组织的
include/
├── aclshmem.h                              // aclshmem所有对外API汇总
├── device/                                 // device侧头文件
│   ├── aclshmem_def.h               // device侧定义的公共标展接口  
│   ├── aclshmemx_def.h               // device侧定义的公共扩展接口  
│   ├── ub2gm/                             // device侧aicore驱动ub2gm数据面低阶接口，扩展接口+x
│   │   ├── aclshmem_device_rma.h           // aicore ub2gm 远端内存访问 RMA 接口
│   │   ├── aclshmem_device_amo.h           // aicore ub2gm 内存原子操作接口
│   │   ├── aclshmem_device_so.h           // aicore ub2gm 信号操作接口
│   │   ├── aclshmem_device_cc.h           // aicore ub2gm 集合通信接口
│   │   ├── aclshmem_device_p2p_sync.h            // aicore ub2gm p2p同步接口
│   │   └── aclshmem_device_mo.h            // aicore ub2gm 内存保序接口
│   │   └── engine/                // aicore 直驱gm2gm 低阶接口，扩展接口+x
│   │       ├── aclshmem_device_mte.h           // aicore 直驱 mte接口
│   │       └── aclshmem_device_tma.h           // aicore 直驱 tma接口
│   ├── gm2gm/                             // device侧aicore驱动gm2gm数据面高阶和低阶接口，扩展接口+x
│   │   ├── aclshmem_device_rma.h           // aicore high-level RMA 接口
│   │   ├── aclshmem_device_amo.h           // aicore high-level 原子内存操作接口
│   │   ├── aclshmem_device_so.h           // aicore high-level 信号操作接口
│   │   ├── aclshmem_device_cc.h           // aicore high-level 集合通信接口
│   │   ├── aclshmem_device_p2p_sync.h     // aicore high-level p2p同步接口
│   │   ├── aclshmem_device_mo.h            // aicore high-level 内存保序接口
│   │   └── engine/                // aicore 直驱gm2gm 低阶接口，扩展接口+x
│   │       ├── aclshmem_device_proxy.h           // aicore -> aicpu 代理
│   │       ├── aclshmem_device_rdma.h           // aicore 直驱 rdma 接口
│   │       ├── aclshmem_device_sdma.h           // aicore 直驱 sdma 接口
│   │       ├── aclshmem_device_ccu.h           // aicore 直驱 ccu 接口           
│   │       └── aclshmem_device_udma.h           // aicore 直驱 udma接口
│   │       └── aclshmem_device_mte.h           // aicore 直驱 udma接口
│   ├── xpu(aicpu、dpu、cx...)/       // device侧aicpu、dpu、CX等驱动gm2gm数据面高阶和低阶接口.以so交付，接入鲲鹏；以so交付的都参考此目录实现交付
│   │   ├── aclshmem_device_rma.h     // xpu high-level RMA接口，扩展接口+x
│   │   ├── aclshmem_device_amo.h     // xpu high-level 内存原子操作接口
│   │   ├── aclshmem_device_so.h      // xpu high-level 接口
│   │   ├── aclshmem_device_cc.h       // xpu high-level 接口
│   │   ├── aclshmem_device_p2p_sync.h   // xpu high-level 接口
│   │   ├── aclshmem_device_mo.h            // xpu high-level 接口
│   │   └── engine/                // xpu 驱动低阶接口
│   │       ├── aclshmem_device_rdma.h           // aicore rdma: aclshmemx_rdma_put()..
│   │       ├── aclshmem_device_sdma.h           // aicore远端内存访问(Remote Memory Access)标准接口
│   │       ├── aclshmem_device_ccu.h           // aicore远端内存访问(Remote Memory Access)标准接口                              // 6类12个头文件
│   │       └── aclshmem_device_udma.h           // aicore udma接口
│   ├── team/                               // device侧通信域管理
│   └── utils/                              // device侧dfx接口
│       ├── aclshmem_log.h                     // device dfx-log接口
│       ├── aclshmem_counter.h                 // device dfx-counter接口
│       ├── aclshmem_profiling.h               // device dfx-profiling接口
│       └── aclshmem_check.h                   // device dfx-check接口  
├── device_host/                            // device_host共用目录
│       └── aclshmem_common_types.h                   // 公共数据结构等 
└── host/                                   // host侧头文件
    ├── aclshmem_host_def.h                    // host侧定义的接口
    ├── init/                               // host侧初始化接口
    ├── team/                               // host侧通信域管理接口
    ├── mem/                                // host侧内存管理接口
    ├── multi_instance/                     // host侧多实例管理（预留）
    ├── data_plane/                         // host侧CPU驱动数据面接口
    │   ├── aclshmem_host_rma.h           // host 接口
    │   ├── aclshmem_host_amo.h           // host 接口
    │   ├── aclshmem_host_so.h           // host 接口
    │   ├── aclshmem_host_cc.h           // host 接口
    │   ├── aclshmem_host_p2p_sync.h            // host high-level 接口
    │   └── aclshmem_host_mo.h            // host high-level 接口
    └── utils/                              // host侧dfx接口
        ├── aclshmem_log.h                     // host dfx-log接口
        ├── aclshmem_counter.h                 // host dfx-counter接口
        ├── aclshmem_profiling.h               // host dfx-profiling接口
        └── aclshmem_check.h                   // host dfx-check接口  

```   
## src
```
|── src
|    |── device             // device侧接口实现
|    |── host           
│    │    ├─common          // host侧通用接口实现、如日志模块
│    │    ├─init            // host侧初始化接口实现
│    │    ├─mem             // host侧内存管理接口实现
│    │    ├─python_wrapper  // Py接口封装
│    │    ├─team            // host侧通信域管理接口实现
|    └── transport          // 建链相关内容
```
## examples
```
├─examples
│  ├─helloworld         // aclshmem简易调用示例
│  └─matmul_allreduce   // 通算融合算子实现样例
```
## tests
```
└─tests
    └─unittest
        ├─init  // 初始化接口单元测试
        ├─mem   // 内存管理接口单元测试
        ├─sync  // 同步管理接口单元测试
        └─team  // 通信域管理接口单元测试
```
## docs
```
├─docs
│    ├─api_demo.md              // aclshmem api调用demo
│    ├─code_organization.md     // 工程组织架构（本文件）
│    ├─example.md               // 使用样例
│    ├─quickstart.md            // 快速开始
│    ├─related_scripts.md       // 相关脚本介绍
|    ├─pythonAPI.md             // aclshmem python api列表
│    └─Troubleshooting_FAQs.md  // QA
```

## scripts
存放相关脚本。  
[脚本具体功能和使用](related_scripts.md)