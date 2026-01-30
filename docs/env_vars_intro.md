# SHMEM Env Vars

## 初始化相关
使用unique id的接口初始化时，需要手动配置环境变量ACLSHMEM_UID_SESSION_ID或者ACLSHMEM_UID_SOCK_IFNAM，同时配置时只读ACLSHMEM_UID_SESSION_ID。指定ACLSHMEM_UID_SESSION_ID时需保证ip可连通，port空闲。指定ACLSHMEM_UID_SOCK_IFNAM时需保证网口指定的网络协议地址存在。

* `ACLSHMEM_UID_SESSION_ID`:直接指定PE 0的监听socket的ip和端口。
ACLSHMEM_UID_SESSION_ID配置示例：
ACLSHMEM_UID_SESSION_ID=127.0.0.1:1234
ACLSHMEM_UID_SESSION_ID=[6666:6666:6666:6666:6666:6666:6666:6666]:886

* `ACLSHMEM_UID_SOCK_IFNAME`:指定PE 0的监听socket的网口名和网络层协议。
ACLSHMEM_UID_SOCK_IFNAM配置示例：
ACLSHMEM_UID_SOCK_IFNAM=enpxxxx:inet4  取ipv4
ACLSHMEM_UID_SOCK_IFNAM=enpxxxx:inet6  取ipv6

## 日志相关
ACLSHMEM提供日志功能协助调测，默认日志等级`error`，落盘到`${HOME}/shmem/log`。课根据下述环境变量控制日志等级，打屏/落盘，落盘路径。

* `ACLSHMEM_LOG_LEVEL`: 用于设置SHMEM日志等级，严重程度从高到低有TRACE、DEBUG、WARN、INFO、ERROR、FATAL。默认为ERROR，调试时建议设置为DEBUG或INFO。
* `ACLSHMEM_LOG_TO_STDOUT`: 用于设置SHMEM日志是否输出到控制台, 0: 关闭, 1: 开启输出到控制台。默认关闭，不输出到控制台，日志会存储到默认路径或指定路径。如果开启则会在控制台打印日志，日志将不再落盘到文件。
* `ACLSHMEM_LOG_PATH`: 用于设置SHMEM日志保存路径， 需要传递合法路径。如不设置默认存储路径为`${HOME}/shmem/log`。

日志详细介绍见[log](log_debug.md)