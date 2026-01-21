## 概述

本样例基于SHMEM工程，介绍了put & get scalar数据传输接口访问Host内存的使用。

## 支持的产品型号

- Atlas A3 训练系列产品/Atlas A3 推理系列产品

## 样例实现

- 算子实现

  本样例呈现的是SHMEM的put & get scalar数据传输接口的使用流程，以下简称put & get接口。

  - 测试用例实现

  （1）初始化[ACL](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/API/appdevgapi/aclcppdevg_03_1945.html)和[SHMEM](../../README.md)，分配input和output数据内存大小并初始化数据，input初始化数据为0，output数据为当前my_pe，后续put接口会把本PE的PE编号发送给下一个PE的input，get接口会获取下一个PE的output数据。

  （2）调用run_demo_scalar启动内核，执行对应kernel实现，前后调用aclshmem_barrier插入同步确保内核执行不受影响。

  （3）执行结果校验，判断各个PE上的结果是否符合预期。

  （4）清理释放SHMEM和ACL相关资源。

  - kernel实现

    （1）kernel侧获取本PE编号、总PE数量、目标PE编号。

    （2）调用aclshmem_int32_p接口向下一个PE的input发送本PE的PE编号，调用aclshmem_quiet插入同步等待scalar数据发送完成。

    （3）调用aclshmem_int32_g获取下一个PE的output数据，调用aclshmem_quiet插入同步等待scalar数据接收完成，并将数据填入本PE的output。

## 编译执行

环境配置请参考[快速上手](../../docs/quickstart.md)。完成环境配置后，执行如下命令可进行功能验证。

```
#执行编译
bash scripts/build.sh -examples -cann
cd examples/rma_d2h_demo
#运行用例
bash run.sh
```

用例执行完成，打屏信息出现“[INFO] demo run end in pe <my_pe>”，说明样例执行结束；打屏信息出现“[SUCCESS] run success in pe <my_pe>”，说明样例执行成功且结果准确。

