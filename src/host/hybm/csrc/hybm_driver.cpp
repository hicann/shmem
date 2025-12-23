/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <fstream>
#include <string>
#include <cstdint>
#include "hybm_logger.h"
#include "shmemi_file_util.h"
#include "hybm.h"

std::string GetDriverVersionPath(const std::string &driverEnvStr, const std::string &keyStr)
{
    std::string driverVersionPath;
    std::string tempPath; // 存放临时路径
    // 查找driver安装路径
    for (uint32_t i = 0; i < driverEnvStr.length(); ++i) {
        // 环境变量中存放的每段路径之间以':'隔开
        if (driverEnvStr[i] != ':') {
            tempPath += driverEnvStr[i];
        }
        // 对存放driver版本文件的路径进行搜索
        if (driverEnvStr[i] == ':' || i == driverEnvStr.length() - 1) {
            if (!shm::utils::FileUtil::Realpath(tempPath)) {
                tempPath.clear();
                continue;
            }
            auto found = tempPath.find(keyStr);
            if (found == std::string::npos) {
                tempPath.clear();
                continue;
            }
            // 确保不是部分匹配
            if (tempPath.length() <= found + keyStr.length() || tempPath[found + keyStr.length()] == '/') {
                driverVersionPath = tempPath.substr(0, found);
                break;
            }
            tempPath.clear();
        }
    }
    return driverVersionPath;
}

std::string LoadDriverVersionInfoFile(const std::string &realName, const std::string &keyStr)
{
    std::string driverVersion;
    // 打开该文件前，判断该文件路径是否有效、规范
    char realFile[shm::utils::FileUtil::GetSafePathMax()] = {0};
    if (shm::utils::FileUtil::IsSymlink(realName) || realpath(realName.c_str(), realFile) == nullptr) {
        BM_LOG_WARN("driver version path is not a valid real path");
        return "";
    }

    // realFile转str,然后open这个str
    std::ifstream infile(realFile, std::ifstream::in);
    if (!infile.is_open()) {
        BM_LOG_WARN("driver version file does not exist");
        return "";
    }

    // 逐行读取，结果放在line中，寻找带有keyStr的字符串
    std::string line;
    int32_t maxRows = 100; // 在文件中读取的最长行数为100，避免超大文件长时间读取
    while (getline(infile, line)) {
        --maxRows;
        if (maxRows < 0) {
            BM_LOG_WARN("driver version file content is too long.");
            infile.close();
            return "";
        }
        auto found = line.find(keyStr);
        // 刚好匹配前缀
        if (found == 0) {
            uint32_t len = line.length() - keyStr.length(); // 版本字符串长度
            driverVersion = line.substr(keyStr.length(), len); // 从keyStr截断
            break;
        }
    }
    infile.close();
    return driverVersion;
}

std::string CastDriverVersion(const std::string &driverEnv)
{
    std::string driverVersionPath = GetDriverVersionPath(driverEnv, "/driver/lib64");
    if (!driverVersionPath.empty()) {
        driverVersionPath += "/driver/version.info";
        std::string driverVersion = LoadDriverVersionInfoFile(driverVersionPath, "Innerversion=");
        return driverVersion;
    } else {
        BM_LOG_WARN("cannot found version file in :" << driverEnv << ", try local default path.");
        driverVersionPath = "/usr/local/Ascend/driver/version.info";  // try default path
        std::string driverVersion = LoadDriverVersionInfoFile(driverVersionPath, "Innerversion=");
        return driverVersion;
    }
}

int32_t GetValueFromVersion(const std::string &ver, std::string key)
{
    int32_t val = 0;
    auto found = ver.find(key);
    if (found == std::string::npos) {
        return -1;
    }

    std::string tmp;
    found += 1;
    while (found < ver.length() && std::isdigit(ver[found])) {
        tmp += ver[found];
        ++found;
    }

    try {
        val = std::stoi(tmp);
    } catch (...) {
        val = -1;
    }
    return val;
}