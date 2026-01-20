/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEM_FABRIC_HYBRID_HYBM_DEFINE_H
#define MEM_FABRIC_HYBRID_HYBM_DEFINE_H

#include <netinet/in.h>
#include <cstdint>
#include <cstddef>

namespace shm {
namespace hybm {

constexpr uint64_t DEVICE_LARGE_PAGE_SIZE = 2UL * 1024UL * 1024UL;  // 大页的size, 2M
constexpr uint64_t HYBM_DEVICE_VA_START = 0x100000000000UL;         // NPU上的地址空间起始: 16T
constexpr uint64_t HYBM_DEVICE_VA_SIZE = 0x80000000000UL;           // NPU上的地址空间范围: 8T
constexpr uint64_t HYBM_DEVICE_VA_END = HYBM_DEVICE_VA_START + HYBM_DEVICE_VA_SIZE;
constexpr uint64_t SVM_END_ADDR = HYBM_DEVICE_VA_START + HYBM_DEVICE_VA_SIZE - (1UL << 30UL); // svm的结尾虚拟地址
constexpr uint64_t HYBM_DEVICE_PRE_META_SIZE = 128UL; // 128B
constexpr uint64_t HYBM_DEVICE_GLOBAL_META_SIZE = HYBM_DEVICE_PRE_META_SIZE; // 128B
constexpr uint64_t HYBM_ENTITY_NUM_MAX = 511UL; // entity最大数量
constexpr uint64_t HYBM_DEVICE_META_SIZE = HYBM_DEVICE_PRE_META_SIZE * HYBM_ENTITY_NUM_MAX
    + HYBM_DEVICE_GLOBAL_META_SIZE; // 64K

constexpr uint64_t HYBM_DEVICE_USER_CONTEXT_PRE_SIZE = 64UL * 1024UL; // 64K
constexpr uint64_t HYBM_DEVICE_INFO_SIZE = HYBM_DEVICE_USER_CONTEXT_PRE_SIZE * HYBM_ENTITY_NUM_MAX
    + HYBM_DEVICE_META_SIZE; // 元数据+用户context,总大小32M, 对齐DEVICE_LARGE_PAGE_SIZE
constexpr uint64_t HYBM_DEVICE_META_ADDR = SVM_END_ADDR - HYBM_DEVICE_INFO_SIZE;
constexpr uint64_t HYBM_DEVICE_USER_CONTEXT_ADDR = HYBM_DEVICE_META_ADDR + HYBM_DEVICE_META_SIZE;
constexpr uint32_t ACL_MEMCPY_HOST_TO_HOST = 0;
constexpr uint32_t ACL_MEMCPY_HOST_TO_DEVICE = 1;
constexpr uint32_t ACL_MEMCPY_DEVICE_TO_HOST = 2;
constexpr uint32_t ACL_MEMCPY_DEVICE_TO_DEVICE = 3;

constexpr uint32_t HCCP_SOCK_CONN_TAG_SIZE = 192;
constexpr uint32_t HCCP_MAX_INTERFACE_NAME_LEN = 256;

constexpr uint64_t EXPORT_INFO_MAGIC = 0xAABB1234FFFFEEEEUL;
constexpr uint64_t EXPORT_SLICE_MAGIC = 0xAABB1234FFFFBBBBUL;
constexpr uint64_t ENTITY_EXPORT_INFO_MAGIC = 0xAABB1234FFFFEE00UL;
constexpr uint64_t HBM_SLICE_EXPORT_INFO_MAGIC = 0xAABB1234FFFFEE01UL;
constexpr uint64_t DRAM_SLICE_EXPORT_INFO_MAGIC = 0xAABB1234FFFFEE02UL;
constexpr uint64_t EXPORT_INFO_VERSION = 0x1UL;

constexpr uint64_t DEVMM_HEAP_SIZE = (1UL << 30UL);     // same definition in devmm

constexpr uint64_t HYBM_HOST_CONN_START_ADDR = 0x30000000000UL;  // 48T
constexpr uint64_t HYBM_HOST_CONN_ADDR_SIZE = 0x100000000000UL;  // 16T

constexpr uint64_t HYBM_GVM_START_ADDR = 0x280000000000UL;      // 40T
constexpr uint64_t HYBM_GVM_END_ADDR = 0xA80000000000UL;        // 168T

inline bool IsVirtualAddressNpu(uint64_t address)
{
    return (address >= HYBM_DEVICE_VA_START && address < (HYBM_DEVICE_VA_START + HYBM_DEVICE_VA_SIZE));
}

inline bool IsVirtualAddressNpu(const void *address)
{
    return IsVirtualAddressNpu(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address)));
}

inline uint64_t Valid48BitsAddress(uint64_t address)
{
    return address & 0xffffffffffffUL;
}

inline const void *Valid48BitsAddress(const void *address)
{
    uint64_t addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address));
    return reinterpret_cast<const void *>(static_cast<uintptr_t>(Valid48BitsAddress(addr)));
}

inline void *Valid48BitsAddress(void *address)
{
    uint64_t addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address));
    return reinterpret_cast<void *>(static_cast<uintptr_t>(Valid48BitsAddress(addr)));
}

enum AscendSocType {
    ASCEND_UNKNOWN = 0,
    ASCEND_910B,
    ASCEND_910C,
};

enum DeviceSystemInfoType {
    INFO_TYPE_PHY_CHIP_ID = 18,
    INFO_TYPE_PHY_DIE_ID,
    INFO_TYPE_SDID = 26,
    INFO_TYPE_SERVER_ID,
    INFO_TYPE_SCALE_TYPE,
    INFO_TYPE_SUPER_POD_ID,
    INFO_TYPE_ADDR_MODE,
};

struct HybmDeviceGlobalMeta {
    uint64_t entityCount;
    uint64_t reserved[15]; // total 128B, equal HYBM_DEVICE_PRE_META_SIZE
};

struct HybmDeviceMeta {
    uint32_t entityId;
    uint32_t rankId;
    uint32_t rankSize;
    uint32_t extraContextSize;
    uint64_t symmetricSize;
    uint64_t qpInfoAddress;
    uint64_t reserved[12]; // total 128B, equal HYBM_DEVICE_PRE_META_SIZE
};

struct HccpRaInitConfig {
    uint32_t phyId;       /**< physical device id */
    uint32_t nicPosition; /**< reference to HccpNetworkMode */
    int hdcType;          /**< reference to drvHdcServiceType */
};

/**
 * @ingroup libinit
 * ip address
 */
union HccpIpAddr {
    struct in_addr addr;
    struct in6_addr addr6;
};

struct HccpRdevInitInfo {
    int mode;
    uint32_t notifyType;
    bool enabled910aLite;    /**< true will enable 910A lite, invalid if enabled_2mb_lite is false; default is false */
    bool disabledLiteThread; /**< true will not start lite thread, flag invalid if enabled_910a/2mb_lite is false */
    bool enabled2mbLite;     /**< true will enable 2MB lite(include 910A & 910B), default is false */
};

/**
 * @ingroup libinit
 * hccp operating environment
 */
enum HccpNetworkMode {
    NETWORK_PEER_ONLINE = 0, /**< Third-party online mode */
    NETWORK_OFFLINE,         /**< offline mode */
    NETWORK_ONLINE,          /**< online mode */
};

/**
 * @ingroup librdma
 * Flag of mr access
 */
enum HccpMrAccessFlags {
    RA_ACCESS_LOCAL_WRITE = 1,         /**< mr local write access */
    RA_ACCESS_REMOTE_WRITE = (1 << 1), /**< mr remote write access */
    RA_ACCESS_REMOTE_READ = (1 << 2),  /**< mr remote read access */
    RA_ACCESS_REDUCE = (1 << 8),
};

enum HccpNotifyType {
    NO_USE = 0,
    NOTIFY = 1,
    EVENTID = 2,
};

/**
 * @ingroup libsocket
 * struct of the client socket
 */
struct HccpSocketConnectInfo {
    void *handle;                      /**< socket handle */
    HccpIpAddr remoteIp;               /**< IP address of remote socket, [0-7] is reserved for vnic */
    uint16_t port;                     /**< Socket listening port number */
    char tag[HCCP_SOCK_CONN_TAG_SIZE]; /**< tag must ended by '\0' */
};

/**
 * @ingroup libsocket
 * Details about socket after socket is linked
 */
struct HccpSocketCloseInfo {
    void *handle; /**< socket handle */
    void *fd;     /**< fd handle */
    int linger;   /**< 0:use(default l_linger is RS_CLOSE_TIMEOUT), others:disuse */
};

/**
 * @ingroup libsocket
 * struct of the listen info
 */
struct HccpSocketListenInfo {
    void *handle;       /**< socket handle */
    unsigned int port;  /**< Socket listening port number */
    unsigned int phase; /**< refer to enum listen_phase */
    unsigned int err;   /**< errno */
};

/**
 * @ingroup libsocket
 * Details about socket after socket is linked
 */
struct HccpSocketInfo {
    void *handle;                      /**< socket handle */
    void *fd;                          /**< fd handle */
    HccpIpAddr remoteIp;               /**< IP address of remote socket */
    int status;                        /**< socket status:0 not connected 1:connected 2:connect timeout 3:connecting */
    char tag[HCCP_SOCK_CONN_TAG_SIZE]; /**< tag must ended by '\0' */
};

/**
 * @ingroup libinit
 * hccp init info
 */
struct HccpRdev {
    uint32_t phyId; /**< physical device id */
    int family;     /**< AF_INET(ipv4) or AF_INET6(ipv6) */
    HccpIpAddr localIp;
};

struct HccpRaGetIfAttr {
    uint32_t phyId;       /**< physical device id */
    uint32_t nicPosition; /**< reference to network_mode */
    bool isAll; /**< valid when nic_position is NETWORK_OFFLINE. false: get specific rnic ip, true: get all rnic ip */
};

struct HccpIfaddrInfo {
    HccpIpAddr ip;       /* Address of interface */
    struct in_addr mask; /* Netmask of interface */
    struct in6_addr maskv6; /* Ipv6 Netmask of interface */
};

struct HccpInterfaceInfo {
    int family;
    int scopeId;
    HccpIfaddrInfo ifaddr;                    /* Address and netmask of interface */
    char ifname[HCCP_MAX_INTERFACE_NAME_LEN]; /* Name of interface */
};

struct HccpSocketWhiteListInfo {
    HccpIpAddr remoteIp;               /**< IP address of remote */
    uint32_t connLimit;                /**< limit of whilte list */
    char tag[HCCP_SOCK_CONN_TAG_SIZE]; /**< tag used for whitelist must ended by '\0' */
};

struct HccpMrInfo {
    void *addr;              /**< starting address of mr */
    unsigned long long size; /**< size of mr */
    int access;              /**< access of mr, reference to HccpMrAccessFlags */
    unsigned int lkey;       /**< local addr access key */
    unsigned int rkey;       /**< remote addr access key */
};

struct HccpCqExtAttr {
    int sendCqDepth;
    int recvDqDepth;
    int sendCqCompVector;
    int recvCqCompVector;
};

enum ibv_qp_type {
    IBV_QPT_RC = 2,
    IBV_QPT_UC,
    IBV_QPT_UD,
    IBV_QPT_RAW_PACKET = 8,
    IBV_QPT_XRC_SEND = 9,
    IBV_QPT_XRC_RECV,
    IBV_QPT_DRIVER = 0xff,
};

struct ibv_qp_cap {
    uint32_t max_send_wr;
    uint32_t max_recv_wr;
    uint32_t max_send_sge;
    uint32_t max_recv_sge;
    uint32_t max_inline_data;
};

struct ibv_qp_init_attr {
    void *qp_context;
    struct ibv_cq *send_cq;
    struct ibv_cq *recv_cq;
    struct ibv_srq *srq;
    struct ibv_qp_cap cap;
    enum ibv_qp_type qp_type;
    int sq_sig_all;
};

union ai_data_plane_cstm_flag {
    struct {
        uint32_t cq_cstm : 1;  // 0: hccp poll cq; 1: caller poll cq
        uint32_t reserved : 31;
    } bs;
    uint32_t value;
};

struct HccpQpExtAttrs {
    int qpMode;
    // cq attr
    HccpCqExtAttr cqAttr;
    // qp attr
    struct ibv_qp_init_attr qp_attr;
    // version control and reserved
    int version;
    int mem_align;  // 0,1:4KB, 2:2MB
    uint32_t udp_sport;
    union ai_data_plane_cstm_flag data_plane_flag;  // only valid in ra_ai_qp_create
    uint32_t reserved[29];
};

struct ai_data_plane_wq {
    unsigned wqn;
    unsigned long long buf_addr;
    unsigned int wqebb_size;
    unsigned int depth;
    unsigned long long head_addr;
    unsigned long long tail_addr;
    unsigned long long swdb_addr;
    unsigned long long db_reg;
    unsigned int reserved[8U];
};

struct ai_data_plane_cq {
    unsigned int cqn;
    unsigned long long buf_addr;
    unsigned int cqe_size;
    unsigned int depth;
    unsigned long long head_addr;
    unsigned long long tail_addr;
    unsigned long long swdb_addr;
    unsigned long long db_reg;
    unsigned int reserved[2U];
};

struct ai_data_plane_info {
    struct ai_data_plane_wq sq;
    struct ai_data_plane_wq rq;
    struct ai_data_plane_cq scq;
    struct ai_data_plane_cq rcq;
    unsigned int reserved[8U];
};

struct HccpAiQpInfo {
    unsigned long long aiQpAddr;  // refer to struct ibv_qp *
    unsigned int sqIndex;         // index of sq
    unsigned int dbIndex;         // index of db

    // below cq related info valid when data_plane_flag.bs.cq_cstm was 1
    unsigned long long ai_scq_addr;  // refer to struct ibv_cq *scq
    unsigned long long ai_rcq_addr;  // refer to struct ibv_cq *rcq
    struct ai_data_plane_info data_plane_info;
};

enum class DBMode : int32_t { INVALID_DB = -1, HW_DB = 0, SW_DB };

struct AiQpRMAWQ {
    uint32_t wqn{0};
    uint64_t bufAddr{0};
    uint32_t wqeSize{0};
    uint32_t depth{0};
    uint64_t headAddr{0};
    uint64_t tailAddr{0};
    DBMode dbMode{DBMode::INVALID_DB};  // 0-hw/1-sw
    uint64_t dbAddr{0};
    uint32_t sl{0};
};

struct AiQpRMACQ {
    uint32_t cqn{0};
    uint64_t bufAddr{0};
    uint32_t cqeSize{0};
    uint32_t depth{0};
    uint64_t headAddr{0};
    uint64_t tailAddr{0};
    DBMode dbMode{DBMode::INVALID_DB};  // 0-hw/1-sw
    uint64_t dbAddr{0};
};

struct RdmaMemRegionInfo {
    uint64_t size{0};  // size of the memory region
    uint64_t addr{0};  // start address of the memory region
    uint32_t lkey{0};
    uint32_t rkey{0};   // key of the memory region
};

struct AiQpRMAQueueInfo {
    uint32_t count;
    struct AiQpRMAWQ *sq;
    struct AiQpRMAWQ *rq;
    struct AiQpRMACQ *scq;
    struct AiQpRMACQ *rcq;
    RdmaMemRegionInfo *mr;
};

// macro for gcc optimization for prediction of if/else
#ifndef LIKELY
#define LIKELY(x) (__builtin_expect(!!(x), 1) != 0)
#endif

#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0) != 0)
#endif

#define HYBM_API __attribute__((visibility("default")))

#define DL_LOAD_SYM(TARGET_FUNC_VAR, TARGET_FUNC_TYPE, FILE_HANDLE, SYMBOL_NAME)                      \
    do {                                                                                              \
        TARGET_FUNC_VAR = (TARGET_FUNC_TYPE)dlsym(FILE_HANDLE, SYMBOL_NAME);                          \
        if ((TARGET_FUNC_VAR) == nullptr) {                                                           \
            BM_LOG_ERROR("Failed to call dlsym to load " << (SYMBOL_NAME) << ", error" << dlerror()); \
            dlclose(FILE_HANDLE);                                                                     \
            return BM_DL_FUNCTION_FAILED;                                                             \
        }                                                                                             \
    } while (0)


#define DL_LOAD_SYM_ALT(TARGET_FUNC_VAR, TARGET_FUNC_TYPE, FILE_HANDLE, SYMBOL_NAME, SYMBOL_NAME_ALT) \
    do {                                                                                              \
        TARGET_FUNC_VAR = (TARGET_FUNC_TYPE)dlsym(FILE_HANDLE, SYMBOL_NAME);                          \
        if ((TARGET_FUNC_VAR) != nullptr) {                                                           \
            BM_LOG_DEBUG("Loaded symbol " << (SYMBOL_NAME) << " successfully");                       \
            break;                                                                                    \
        }                                                                                             \
        TARGET_FUNC_VAR = (TARGET_FUNC_TYPE)dlsym(FILE_HANDLE, SYMBOL_NAME_ALT);                      \
        if ((TARGET_FUNC_VAR) != nullptr) {                                                           \
            BM_LOG_DEBUG("Loaded symbol " << (SYMBOL_NAME_ALT) << " successfully");                   \
            break;                                                                                    \
        }                                                                                             \
        BM_LOG_ERROR("Failed to call dlsym to load " << (SYMBOL_NAME) << " or " << (SYMBOL_NAME_ALT)  \
                                                     << ", error" << dlerror());                      \
        dlclose(FILE_HANDLE);                                                                         \
        FILE_HANDLE = nullptr;                                                                        \
        return BM_DL_FUNCTION_FAILED;                                                                 \
    } while (0)

enum HybmGvaVersion : uint32_t {
    HYBM_GVA_V1 = 0,
    HYBM_GVA_V2 = 1,
    HYBM_GVA_V3 = 2,
    HYBM_GVA_UNKNOWN
};

typedef struct drv_mem_handle {
    int id;
    uint32_t side;
    uint32_t sdid;
    uint32_t devid;
    uint32_t module_id;
    uint64_t pg_num;

    uint32_t pg_type;
    uint32_t phy_mem_type;
} drv_mem_handle_t;

typedef enum {
    MEM_HANDLE_TYPE_NONE = 0x0,
    MEM_HANDLE_TYPE_FABRIC = 0x1,
    MEM_HANDLE_TYPE_MAX = 0x2,
} drv_mem_handle_type;

#define MEM_SHARE_HANDLE_LEN 128
struct MemShareHandle {
    uint8_t share_info[MEM_SHARE_HANDLE_LEN];
};

enum drv_mem_side {
    MEM_HOST_SIDE = 0,
    MEM_DEV_SIDE,
    MEM_HOST_NUMA_SIDE,
    MEM_MAX_SIDE
};

enum drv_mem_pg_type {
    MEM_NORMAL_PAGE_TYPE = 0,
    MEM_HUGE_PAGE_TYPE,
    MEM_GIANT_PAGE_TYPE,
    MEM_MAX_PAGE_TYPE
};

enum drv_mem_type {
    MEM_HBM_TYPE = 0,
    MEM_DDR_TYPE,
    MEM_P2P_HBM_TYPE,
    MEM_P2P_DDR_TYPE,
    MEM_TS_DDR_TYPE,
    MEM_MAX_TYPE
};

struct drv_mem_prop {
    uint32_t side;       /* enum drv_mem_side */
    uint32_t devid;
    uint32_t module_id;  /* module id defines in ascend_hal_define.h */

    uint32_t pg_type;    /* enum drv_mem_pg_type */
    uint32_t mem_type;   /* enum drv_mem_type */
    uint64_t reserve;
};

typedef enum {
    MEM_ALLOC_GRANULARITY_MINIMUM = 0x0,
    MEM_ALLOC_GRANULARITY_RECOMMENDED,
    MEM_ALLOC_GRANULARITY_INVALID,
} drv_mem_granularity_options;

enum ShareHandleAttrType {
    SHR_HANDLE_ATTR_NO_WLIST_IN_SERVER = 0,
    SHR_HANDLE_ATTR_TYPE_MAX
};

#define SHR_HANDLE_WLIST_ENABLE     0x0
#define SHR_HANDLE_NO_WLIST_ENABLE  0x1
struct ShareHandleAttr {
    unsigned int enableFlag;     /* wlist enable: 0 no wlist enable: 1 */
    unsigned int rsv[8];
};

#define MEM_RSV_TYPE_DEVICE_SHARE_BIT   8 /* mmap va in all opened devices, exclude host to avoid va conflict */
#define MEM_RSV_TYPE_DEVICE_SHARE       (0x1u << MEM_RSV_TYPE_DEVICE_SHARE_BIT)
#define MEM_RSV_TYPE_REMOTE_MAP_BIT     9 /* this va only map remote addr, not create double page table */
#define MEM_RSV_TYPE_REMOTE_MAP         (0x1u << MEM_RSV_TYPE_REMOTE_MAP_BIT)

}  // namespace hybm
}  // namespace shm

#endif  // MEM_FABRIC_HYBRID_HYBM_DEFINE_H
