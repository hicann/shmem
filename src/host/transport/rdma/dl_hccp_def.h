/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DL_HCCP_DEF_H
#define DL_HCCP_DEF_H

#include <sys/types.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <net/if.h>
#include <ifaddrs.h>
#include <cstdint>
#include <cstddef>
#include <ostream>
#include <string>
#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>
#include "aclshmem.h"
#include "utils/aclshmemi_functions.h"
#include "utils/aclshmemi_host_types.h"

using Result = int32_t;

constexpr uint32_t HCCL_ROOT_INFO_BYTES = 256;  // 4108: root info length
constexpr uint32_t HCCP_SOCK_CONN_TAG_SIZE = 192;
constexpr uint32_t HCCP_MAX_INTERFACE_NAME_LEN = 256;

/**
 * @brief HCCL root info
 */
struct HcclRootInfo {
    char internal[HCCL_ROOT_INFO_BYTES];
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

inline std::ostream &operator<<(std::ostream &output, const HccpSocketConnectInfo &info)
{
    output << "HccpSocketConnectInfo(socketHandle=" << info.handle << ", remoteIp=" << inet_ntoa(info.remoteIp.addr)
           << ", port=" << info.port << ")";
    return output;
}

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

enum ibv_wc_status {
    IBV_WC_SUCCESS,
    IBV_WC_LOC_LEN_ERR,
    IBV_WC_LOC_QP_OP_ERR,
    IBV_WC_LOC_EEC_OP_ERR,
    IBV_WC_LOC_PROT_ERR,
    IBV_WC_WR_FLUSH_ERR,
    IBV_WC_MW_BIND_ERR,
    IBV_WC_BAD_RESP_ERR,
    IBV_WC_LOC_ACCESS_ERR,
    IBV_WC_REM_INV_REQ_ERR,
    IBV_WC_REM_ACCESS_ERR,
    IBV_WC_REM_OP_ERR,
    IBV_WC_RETRY_EXC_ERR,
    IBV_WC_RNR_RETRY_EXC_ERR,
    IBV_WC_LOC_RDD_VIOL_ERR,
    IBV_WC_REM_INV_RD_REQ_ERR,
    IBV_WC_REM_ABORT_ERR,
    IBV_WC_INV_EECN_ERR,
    IBV_WC_INV_EEC_STATE_ERR,
    IBV_WC_FATAL_ERR,
    IBV_WC_RESP_TIMEOUT_ERR,
    IBV_WC_GENERAL_ERR
};

enum ibv_wc_opcode {
    IBV_WC_SEND,
    IBV_WC_RDMA_WRITE,
    IBV_WC_RDMA_READ,
    IBV_WC_COMP_SWAP,
    IBV_WC_FETCH_ADD,
    IBV_WC_BIND_MW,
    /*
 * Set value of IBV_WC_RECV so consumers can test if a completion is a
 * receive by testing (opcode & IBV_WC_RECV).
 */
    IBV_WC_RECV = 1 << 7,
    IBV_WC_RECV_RDMA_WITH_IMM
};

struct ibv_wc {
    uint64_t wr_id;
    enum ibv_wc_status status;
    enum ibv_wc_opcode opcode;
    uint32_t vendor_err;
    uint32_t byte_len;
    uint32_t imm_data; /* in network byte order */
    uint32_t qp_num;
    uint32_t src_qp;
    int wc_flags;
    uint16_t pkey_index;
    uint16_t slid;
    uint8_t sl;
    uint8_t dlid_path_bits;
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

/**
 * @ingroup librdma
 * Scatter and gather element
 */
struct sg_list {
    uint64_t addr; /**< address of buf */
    uint32_t len;  /**< len of buf */
    uint32_t lkey; /**< local addr access key */
};

/**
 * @ingroup librdma
 * RDMA work request
 */
struct send_wr {
    struct sg_list *buf_list; /**< list of sg */
    uint16_t buf_num;         /**< num of buf_list */
    uint64_t dst_addr;        /**< destination address */
    uint32_t rkey;            /**< remote address access key */
    uint32_t op;              /**< operations of RDMA supported:RDMA_WRITE:0 */
    int send_flag;            /**< reference to ra_send_flags */
};

/**
 * @ingroup librdma
 * wqe template info
 */
struct wqe_info {
    unsigned int sq_index;  /**< index of sq */
    unsigned int wqe_index; /**< index of wqe */
};

enum ra_send_flags {
    RA_SEND_FENCE = 1 << 0,     /**< RDMA operation with fence */
    RA_SEND_SIGNALED = 1 << 1,  /**< RDMA operation with signaled */
    RA_SEND_SOLICITED = 1 << 2, /**< RDMA operation with solicited */
    RA_SEND_INLINE = 1 << 3,    /**< RDMA operation with inline */
};
/**
 * @ingroup librdma
 * doorbell info
 */
struct db_info {
    unsigned int db_index; /**< index of db */
    unsigned long db_info; /**< db content */
};

/**
 * @ingroup librdma
 * respond of sending work request
 */
struct send_wr_rsp {
    union {
        struct wqe_info wqe_tmp; /**< wqe template info */
        struct db_info db;       /**< doorbell info */
    };
};
/**
 * @brief handle to HCCL communicator
 */
typedef void *HcclComm;

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
            std::cout << "Failed to call dlsym to load symbol" << SYMBOL_NAME << std::endl; \
            dlclose(FILE_HANDLE);                                                                     \
            return -1;                                                                                \
        }                                                                                             \
    } while (0)


inline std::ostream &operator<<(std::ostream &output, const HccpRaInitConfig &config)
{
    output << "HccpRaInitConfig(phyId=" << config.phyId << ", nicPosition=" << config.nicPosition
           << ", hdcType=" << config.hdcType << ")";
    return output;
}

inline std::ostream &operator<<(std::ostream &output, const HccpRdevInitInfo &info)
{
    output << "HccpRdevInitInfo(mode=" << info.mode << ", notify=" << info.notifyType
           << ", enabled910aLite=" << info.enabled910aLite << ", disabledLiteThread=" << info.disabledLiteThread
           << ", enabled2mbLite=" << info.enabled2mbLite << ")";
    return output;
}

inline std::ostream &operator<<(std::ostream &output, const HccpRdev &rdev)
{
    output << "HccpRdev(phyId=" << rdev.phyId << ", family=" << rdev.family
           << ", rdev.ip=" << inet_ntoa(rdev.localIp.addr) << ")";
    return output;
}

struct RegMemResult {
    uint32_t reserved{0};
    uint64_t address{0};
    uint64_t size{0};
    void *mrHandle{nullptr};
    uint32_t lkey{0};
    uint32_t rkey{0};

    RegMemResult() = default;

    RegMemResult(uint64_t addr, uint64_t sz, void *hd, uint32_t lk, uint32_t rk)
        : address(addr),
          size(sz),
          mrHandle(hd),
          lkey(lk),
          rkey(rk)
    {
    }
};

inline std::ostream &operator<<(std::ostream &output, const RegMemResult &mr)
{
    output << "RegMemResult(address = " << mr.address << ", size = " << mr.size
    << ", lkey = " << mr.lkey << ", rkey = " << mr.rkey << ")";
    return output;
}

constexpr int32_t REG_MR_ACCESS_FLAG_LOCAL_WRITE = 0x1;
constexpr int32_t REG_MR_ACCESS_FLAG_REMOTE_WRITE = 0x2;
constexpr int32_t REG_MR_ACCESS_FLAG_REMOTE_READ = 0x4;
constexpr int32_t REG_MR_ACCESS_FLAG_BOTH_READ_WRITE = 0x7;

typedef enum {
    HYBM_ROLE_PEER = 0,
    HYBM_ROLE_SENDER,
    HYBM_ROLE_RECEIVER,
    HYBM_ROLE_BUTT
} hybm_role_type;

struct TransportOptions {
    uint32_t rankId;
    uint32_t rankCount;
    uint32_t protocol;
    hybm_role_type role;
    int nic;
    int32_t dev_id;
    int32_t logic_dev_id;
};

struct TransportMemoryRegion {
    uint64_t addr = 0;  /* virtual address of memory could be hbm or host dram */
    uint64_t size = 0;  /* size of memory to be registered */
    int32_t access = REG_MR_ACCESS_FLAG_BOTH_READ_WRITE; /* access right by local and remote */
    uint32_t flags = 0; /* optional flags: 加一个flag标识是DRAM还是HBM */

    friend std::ostream &operator<<(std::ostream &output, const TransportMemoryRegion &mr)
    {
        output << "MemoryRegion address size=" << mr.size << ", access=" << mr.access
            << ", flags=" << mr.flags << ")";
        return output;
    }
};

using MemoryRegionMap = std::map<uint64_t, RegMemResult, std::greater<uint64_t>>;

struct TransportMemoryKey {
    uint32_t keys[16];

    friend std::ostream &operator<<(std::ostream &output, const TransportMemoryKey &key)
    {
        output << "MemoryKey" << std::hex;
        for (auto i = 0U; i < sizeof(key.keys) / sizeof(key.keys[0]); i++) {
            output << "-" << key.keys[i];
        }
        output << std::dec;
        return output;
    }
};

#define container_of(ptr, type, member)                                              \
    ({                                                                               \
        const typeof(((const type *)0)->member) *__mptr = (ptr);                     \
        (const type *)(const void *)((const char *)__mptr - offsetof(type, member)); \
    })

union RegMemKeyUnion {
    TransportMemoryKey commonKey;
    RegMemResult deviceKey;
};

struct ConnectRankInfo {
    hybm_role_type role;
    sockaddr_in network;
    RegMemResult mr;

    ConnectRankInfo(hybm_role_type r, sockaddr_in nw, RegMemResult memory_region) : role{r}, 
        network{std::move(nw)}, mr{memory_region} {}
};

struct TransportRankPrepareInfo {
    std::string nic;
    hybm_role_type role{HYBM_ROLE_PEER};
    RegMemResult mr;

    TransportRankPrepareInfo() {}

    TransportRankPrepareInfo(std::string n, RegMemResult k)
        : nic{std::move(n)}, role{HYBM_ROLE_PEER}, mr{k} {}

    TransportRankPrepareInfo(std::string n, hybm_role_type r, RegMemResult k)
        : nic{std::move(n)}, role{r}, mr{k} {}

    friend std::ostream &operator<<(std::ostream &output, const TransportRankPrepareInfo &info)
    {
        output << "PrepareInfo(nic=" << info.nic << ", role=" << info.role << ", mr=" << info.mr;
        return output;
    }
};

struct HybmTransPrepareOptions {
    std::unordered_map<uint32_t, TransportRankPrepareInfo> options;

    friend std::ostream &operator<<(std::ostream &output, const HybmTransPrepareOptions &info)
    {
        output << "PrepareOptions(";
        for (auto &op : info.options) {
            output << op.first << " => " << op.second << ", ";
        }
        output << ")";
        return output;
    }
};
#endif  // DL_HCCP_DEF_H