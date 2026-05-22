cmake_minimum_required(VERSION 3.16...3.26)

if(NOT DEFINED SKBUILD_PROJECT_NAME)
  set(SKBUILD_PROJECT_NAME "mmu")
endif()
if(NOT DEFINED SKBUILD_PROJECT_VERSION)
  set(SKBUILD_PROJECT_VERSION "0.1.2")
endif()

string(REPLACE "-" "_" SKBUILD_PROJECT_NAME "${SKBUILD_PROJECT_NAME}")
string(REGEX REPLACE "\\.?[a-z]+[0-9]+" "" SKBUILD_PROJECT_VERSION "${SKBUILD_PROJECT_VERSION}")

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  DESCRIPTION "Fast binary classification metrics."
  LANGUAGES CXX)

set(CMAKE_MODULE_PATH_SAVED ${CMAKE_MODULE_PATH})
list(INSERT CMAKE_MODULE_PATH 0 "${PROJECT_SOURCE_DIR}/src/mmu_core/cmake")

if(NOT SKBUILD)
  message(WARNING
          "This CMake file is primarily intended for scikit-build-core/pip builds. "
          "Use 'pip install -e .' or 'pip install .' for a complete local install.")
endif()

if(MMU_MBUILD)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/src")
endif()

# ##############################################################################
# SETTINGS                                   #
# ##############################################################################
set(MMU_CPP_STANDARD "17" CACHE STRING "C++ standard used for building mmu")
set(CMAKE_CXX_STANDARD ${MMU_CPP_STANDARD})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MMU_DEV_MODE OFF)
option(MMU_ENABLE_DEVMODE OFF)
option(MMU_CICD_MODE OFF)
option(MMU_VALGRIND_MODE OFF)
option(MMU_ENABLE_OPENMP OFF)
option(MMU_DISABLE_OPENMP OFF)
option(MMU_ENABLE_INTERNAL_TESTS OFF)
option(MMU_ENABLE_ARCH_FLAGS OFF)
option(MMU_ENABLE_ARCH_FLAGS_SIMPLE OFF)

if(MMU_ENABLE_DEVMODE)
  set(MMU_DEV_MODE ON)
endif()

set(DEFAULT_BUILD_TYPE Release)
include(BuildType)

if(MMU_DEV_MODE)
  set(MMU_ENABLE_INTERNAL_TESTS ON)
endif()

# ##############################################################################
# DEPENDENCIES                                   #
# ##############################################################################
include(GNUInstallDirs)

find_package(
  Python3
  COMPONENTS Development NumPy
  QUIET)
if(NOT Python3_FOUND)
  find_package(
    Python3
    COMPONENTS Development.Module NumPy
    REQUIRED)
endif()
# Needed to ensure the same Python executable is found by Pybind11
if(NOT DEFINED PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
endif()

find_package(pybind11 CONFIG REQUIRED)
if(MMU_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
elseif(MMU_DISABLE_OPENMP)
  message(STATUS "mmu: OpenMP is disabled")
else()
  find_package(OpenMP)
endif()

# ##############################################################################
# HEADER TARGET                                 #
# ##############################################################################
add_library(mmu_headers INTERFACE)
add_library(mmu::headers ALIAS mmu_headers)
target_include_directories(mmu_headers INTERFACE ${PROJECT_SOURCE_DIR}/src/mmu_core/include/)

# ##############################################################################
# FULL TARGET                                   #
# ##############################################################################
add_library(mmu_mmu INTERFACE)
add_library(mmu::mmu ALIAS mmu_mmu)
target_include_directories(
  mmu_mmu INTERFACE ${PROJECT_SOURCE_DIR}/src/mmu_core/include/ ${Python3_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}/src/mmu_core/external/pcg-cpp/include)

if(MMU_DEV_MODE AND NOT MMU_CICD_MODE)
  target_compile_options(mmu_mmu INTERFACE -Wall -Wextra -Wunused-variable
                                           -Wunused-const-variable)
endif()

target_link_libraries(mmu_mmu INTERFACE pybind11::pybind11 Python3::NumPy)

if(OpenMP_CXX_FOUND)
  target_compile_definitions(mmu_mmu INTERFACE MMU_HAS_OPENMP_SUPPORT=TRUE)
  target_link_libraries(mmu_mmu INTERFACE OpenMP::OpenMP_CXX)
endif()

set(MMU_BINDINGS_PATH "${PROJECT_SOURCE_DIR}/src/mmu_core/src")
set(MMU_API_PATH "${PROJECT_SOURCE_DIR}/src/mmu_core/src/api")

set(MMU_SRC_FILES
    ${MMU_BINDINGS_PATH}/extension.cpp
    ${MMU_BINDINGS_PATH}/utils.cpp
    ${MMU_BINDINGS_PATH}/confusion_matrix.cpp
    ${MMU_API_PATH}/metrics.cpp
    ${MMU_BINDINGS_PATH}/metrics.cpp
    ${MMU_API_PATH}/pr_multn_loglike.cpp
    ${MMU_BINDINGS_PATH}/pr_multn_loglike.cpp
    ${MMU_API_PATH}/roc_multn_loglike.cpp
    ${MMU_BINDINGS_PATH}/roc_multn_loglike.cpp
    ${MMU_API_PATH}/recall_ppn_multn_loglike.cpp
    ${MMU_BINDINGS_PATH}/recall_ppn_multn_loglike.cpp)

pybind11_add_module(_mmu_core MODULE ${MMU_SRC_FILES})
target_link_libraries(_mmu_core PUBLIC mmu::mmu)
target_compile_definitions(_mmu_core PRIVATE EXTENSION_MODULE_NAME=_mmu_core VERSION_INFO=${PROJECT_VERSION})
target_compile_options(_mmu_core PRIVATE "$<$<CONFIG:RELEASE>:${MMU_ARCHITECTURE_FLAGS}>")
set_property(TARGET _mmu_core PROPERTY CXX_STANDARD ${MMU_CPP_STANDARD})
set_property(TARGET _mmu_core PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET _mmu_core PROPERTY POSITION_INDEPENDENT_CODE ON)

if(SKBUILD)
  install(TARGETS _mmu_core LIBRARY DESTINATION "${PROJECT_NAME}/lib")
else()
  install(TARGETS _mmu_core LIBRARY DESTINATION "${PROJECT_SOURCE_DIR}/src/mmu/lib")
endif()

if(MMU_ENABLE_INTERNAL_TESTS)
  add_subdirectory(${PROJECT_SOURCE_DIR}/src/mmu_core/tests)
endif() # MMU_ENABLE_INTERNAL_TESTS

unset(MMU_DEV_MODE CACHE)
unset(MMU_ENABLE_DEVMODE CACHE)
unset(MMU_CICD_MODE CACHE)
unset(MMU_VALGRIND_MODE CACHE)
unset(MMU_ENABLE_OPENMP CACHE)
unset(MMU_DISABLE_OPENMP CACHE)
unset(MMU_ENABLE_INTERNAL_TESTS CACHE)
unset(MMU_ENABLE_ARCH_FLAGS CACHE)
unset(MMU_ENABLE_ARCH_FLAGS_SIMPLE CACHE)
unset(MMU_CPP_STANDARD CACHE)
