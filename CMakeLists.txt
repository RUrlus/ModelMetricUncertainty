cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED MMU_VERSION_INFO)
  set(MMU_VERSION_INFO "0.1.2")
endif()

project(
  mmu
  VERSION ${MMU_VERSION_INFO}
  DESCRIPTION "Fast binary classification metrics."
  LANGUAGES CXX)

set(CMAKE_MODULE_PATH_SAVED ${CMAKE_MODULE_PATH})
list(INSERT CMAKE_MODULE_PATH 0 "${PROJECT_SOURCE_DIR}/cmake")
# ##############################################################################
# SETTINGS                                   #
# ##############################################################################
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MMU_DEV_MODE OFF)
option(MMU_CICD_MODE OFF)
option(MMU_VALGRIND_MODE OFF)
option(MMU_ENABLE_OPENMP OFF)
option(MMU_DISABLE_OPENMP OFF)
option(MMU_ENABLE_INTERNAL_TESTS OFF)
option(MMU_ENABLE_ARCH_FLAGS OFF)
option(MMU_ENABLE_ARCH_FLAGS_SIMPLE OFF)

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
target_include_directories(mmu_headers INTERFACE ${PROJECT_SOURCE_DIR}/include/)

# ##############################################################################
# FULL TARGET                                   #
# ##############################################################################
add_library(mmu_mmu INTERFACE)
add_library(mmu::mmu ALIAS mmu_mmu)
target_include_directories(
  mmu_mmu INTERFACE ${PROJECT_SOURCE_DIR}/include/ ${Python3_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}/external/pcg-cpp/include)

if(MMU_DEV_MODE AND NOT MMU_CICD_MODE)
  target_compile_options(mmu_mmu INTERFACE -Wall -Wextra -Wunused-variable
                                           -Wunused-const-variable)
endif()

target_link_libraries(mmu_mmu INTERFACE pybind11::pybind11 Python3::NumPy)

if(OpenMP_CXX_FOUND)
  target_compile_definitions(mmu_mmu INTERFACE MMU_HAS_OPENMP_SUPPORT=TRUE)
  target_link_libraries(mmu_mmu INTERFACE OpenMP::OpenMP_CXX)
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/src/bindings)

if(MMU_ENABLE_INTERNAL_TESTS)
  add_subdirectory(${PROJECT_SOURCE_DIR}/src/tests)
endif() # MMU_ENABLE_INTERNAL_TESTS

unset(MMU_DEV_MODE CACHE)
unset(MMU_CICD_MODE CACHE)
unset(MMU_VALGRIND_MODE CACHE)
unset(MMU_ENABLE_OPENMP CACHE)
unset(MMU_DISABLE_OPENMP CACHE)
unset(MMU_ENABLE_INTERNAL_TESTS CACHE)
unset(MMU_ENABLE_ARCH_FLAGS CACHE)
unset(MMU_ENABLE_ARCH_FLAGS_SIMPLE CACHE)
