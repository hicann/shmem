# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_subdirectory(host/hybm)
add_subdirectory(host/bootstrap/config_store)

if (BUILD_PYTHON STREQUAL "ON")
    add_subdirectory(host/python_wrapper)
endif ()

set(ACLSHMEM_MPI_SUPPORT OFF)

if(ACLSHMEM_MPI_SUPPORT)
  find_package(MPI REQUIRED)
else()
  find_package(MPI)
  if(MPI_FOUND)
        set(ACLSHMEM_MPI_SUPPORT ON)
  endif()
endif()

file(GLOB_RECURSE ACLSHMEM_KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/device/*.cpp)
add_library(aclshmem_device OBJECT ${ACLSHMEM_KERNEL_FILES})
target_compile_options(aclshmem_device PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220)
target_include_directories(aclshmem_device
        PUBLIC
        ${PROJECT_SOURCE_DIR}/include/
        ${PROJECT_SOURCE_DIR}/src/device/
)

file(GLOB_RECURSE ACLSHMEM_HOST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/host/*.cpp)
list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "python_wrapper")
list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "host/bootstrap")
list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "host/transport")
list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "host/hybm")
list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "host/utils/log")

add_library(aclshmem_host OBJECT ${ACLSHMEM_HOST_FILES})
target_compile_options(aclshmem_host PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(aclshmem_host
        PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src/host
        ${PROJECT_SOURCE_DIR}/src/device
        ${PROJECT_SOURCE_DIR}/src/host/utils
        ${PROJECT_SOURCE_DIR}/src/host/hybm/include
        ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store
        ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/include
)

file(GLOB_RECURSE ACLSHMEM_UTILS_FILES ${CMAKE_CURRENT_SOURCE_DIR}/host/utils/log/*.cpp)
add_library(shmem_utils SHARED ${ACLSHMEM_UTILS_FILES})
target_compile_options(shmem_utils PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(shmem_utils
        PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src/host
)
install(TARGETS shmem_utils
        LIBRARY DESTINATION lib
)

add_library(shmem SHARED $<TARGET_OBJECTS:aclshmem_device> $<TARGET_OBJECTS:aclshmem_host>)
target_link_options(shmem PRIVATE --cce-fatobj-link)
target_link_libraries(shmem PRIVATE shmem_utils)

if(NOT USE_UNIT_TEST)
        target_link_libraries(shmem PRIVATE hybmm_shared config_store_shared)
endif()

# MPI
if(ACLSHMEM_MPI_SUPPORT)
    separate_arguments(ACLSHMEM_CXX_LINK_FLAGS NATIVE_COMMAND "${MPI_CXX_LINK_FLAGS}")
    target_link_options(shmem INTERFACE ${ACLSHMEM_CXX_LINK_FLAGS})
    target_compile_definitions(shmem INTERFACE ${MPI_CXX_COMPILE_DEFINITIONS})
    target_compile_options(shmem INTERFACE ${MPI_CXX_COMPILE_OPTIONS})
    
    add_library(aclshmem_bootstrap_mpi SHARED)
    target_sources(aclshmem_bootstrap_mpi PRIVATE host/bootstrap/shmemi_bootstrap_mpi.cpp)
    target_link_libraries(aclshmem_bootstrap_mpi PRIVATE MPI::MPI_CXX shmem_utils)
    target_include_directories(aclshmem_bootstrap_mpi
                                PRIVATE
                                ${PROJECT_SOURCE_DIR}/include
                                ${PROJECT_SOURCE_DIR}/src/host
                                ${PROJECT_SOURCE_DIR}/src/host/hybm/include
                                ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store
                                ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/include
    )
    set_target_properties(aclshmem_bootstrap_mpi PROPERTIES PREFIX "")
    install(TARGETS aclshmem_bootstrap_mpi
        LIBRARY DESTINATION lib
    )

endif()
# UID
add_library(aclshmem_bootstrap_uid SHARED)

target_sources(aclshmem_bootstrap_uid PRIVATE host/bootstrap/socket/uid_socket.cpp
                                             host/bootstrap/shmemi_bootstrap_uid.cpp)
target_include_directories(aclshmem_bootstrap_uid
                        PRIVATE
                        ${PROJECT_SOURCE_DIR}/include
                        ${PROJECT_SOURCE_DIR}/src/host
                        ${PROJECT_SOURCE_DIR}/src/host/hybm/include
                        ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store
                        ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/include
)
set_target_properties(aclshmem_bootstrap_uid PROPERTIES PREFIX "")
target_link_libraries(aclshmem_bootstrap_uid PRIVATE shmem_utils)
install(TARGETS aclshmem_bootstrap_uid
        LIBRARY DESTINATION lib)

# 安装配置
install(TARGETS shmem
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/device
        DESTINATION src
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/host_device
        DESTINATION src
)
