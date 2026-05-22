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
option(MMU_ENABLE_DEBUG OFF)
option(MMU_CICD_MODE OFF)
option(MMU_VALGRIND_MODE OFF)
option(MMU_ENABLE_OPENMP OFF)
option(MMU_DISABLE_OPENMP OFF)
option(MMU_VENDOR_OPENMP OFF)
option(MMU_ENABLE_INTERNAL_TESTS OFF)
option(MMU_ENABLE_ARCH_FLAGS OFF)

include(ConfigureBuildType)
include(FindDependencies)
include(ConfigureTarget)

if(MMU_ENABLE_INTERNAL_TESTS)
  add_subdirectory(${PROJECT_SOURCE_DIR}/src/mmu_core/tests)
endif() # MMU_ENABLE_INTERNAL_TESTS

include(CleanUp)
