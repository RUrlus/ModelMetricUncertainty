CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

IF (NOT DEFINED MMU_VERSION_INFO)
    SET(MMU_VERSION_INFO "0.1.0")
ENDIF ()

PROJECT(
    mmu
    VERSION ${MMU_VERSION_INFO}
    DESCRIPTION "Fast binary classification metrics."
    LANGUAGES CXX
)

SET(CMAKE_MODULE_PATH_SAVED ${CMAKE_MODULE_PATH})
LIST(INSERT CMAKE_MODULE_PATH 0 "${PROJECT_SOURCE_DIR}/cmake")
################################################################################
#                                   SETTINGS                                   #
################################################################################
SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

OPTION(MMU_DEV_MODE OFF)
OPTION(MMU_CICD_MODE OFF)
OPTION(MMU_VALGRIND_MODE OFF)
OPTION(MMU_ENABLE_OPENMP OFF)
OPTION(MMU_DISABLE_OPENMP OFF)
OPTION(MMU_ENABLE_INTERNAL_TESTS OFF)
OPTION(MMU_ENABLE_ARCH_FLAGS OFF)
OPTION(MMU_ENABLE_ARCH_FLAGS_SIMPLE OFF)

SET(DEFAULT_BUILD_TYPE Release)
INCLUDE(BuildType)

IF (MMU_DEV_MODE)
    SET(MMU_ENABLE_INTERNAL_TESTS ON)
ENDIF()

################################################################################
#                               DEPENDENCIES                                   #
################################################################################
INCLUDE(GNUInstallDirs)

FIND_PACKAGE(Python3 COMPONENTS Development NumPy QUIET)
IF (NOT Python3_FOUND)
    FIND_PACKAGE(Python3 COMPONENTS Development.Module NumPy REQUIRED)
ENDIF()
# Needed to ensure the same Python executable is found by Pybind11
IF (NOT DEFINED PYTHON_EXECUTABLE)
    SET(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
ENDIF ()

FIND_PACKAGE(pybind11 CONFIG REQUIRED)
IF (MMU_ENABLE_OPENMP)
    FIND_PACKAGE(OpenMP REQUIRED)
ELSEIF (MMU_DISABLE_OPENMP)
    message(STATUS "mmu: OpenMP is disabled")
ELSE ()
    FIND_PACKAGE(OpenMP)
ENDIF ()

################################################################################
#                                HEADER TARGET                                 #
################################################################################
ADD_LIBRARY(mmu_headers INTERFACE)
ADD_LIBRARY(mmu::headers ALIAS mmu_headers)
TARGET_INCLUDE_DIRECTORIES(mmu_headers 
    INTERFACE
        ${PROJECT_SOURCE_DIR}/include/
)

################################################################################
#                                FULL TARGET                                   #
################################################################################
ADD_LIBRARY(mmu_mmu INTERFACE)
ADD_LIBRARY(mmu::mmu ALIAS mmu_mmu)
TARGET_INCLUDE_DIRECTORIES(mmu_mmu 
    INTERFACE
        ${PROJECT_SOURCE_DIR}/include/
        ${Python3_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/external/pcg-cpp/include
)

IF(MMU_DEV_MODE AND NOT MMU_CICD_MODE)
    TARGET_COMPILE_OPTIONS(mmu_mmu
        INTERFACE
        -Wall
        -Wextra
        -Wunused-variable
        -Wunused-const-variable
    )
ENDIF()

TARGET_LINK_LIBRARIES(
    mmu_mmu
    INTERFACE
        pybind11::pybind11
        Python3::NumPy
)

IF (OpenMP_CXX_FOUND)
    TARGET_COMPILE_DEFINITIONS(
        mmu_mmu
        INTERFACE
        MMU_HAS_OPENMP_SUPPORT=TRUE
    )
    TARGET_LINK_LIBRARIES(
        mmu_mmu
        INTERFACE
            OpenMP::OpenMP_CXX
    )
ENDIF ()

ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/src/bindings)

IF (MMU_ENABLE_INTERNAL_TESTS)
    ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/src/tests)
ENDIF ()  # MMU_ENABLE_INTERNAL_TESTS

UNSET(MMU_DEV_MODE CACHE)
UNSET(MMU_CICD_MODE CACHE)
UNSET(MMU_VALGRIND_MODE CACHE)
UNSET(MMU_ENABLE_OPENMP CACHE)
UNSET(MMU_DISABLE_OPENMP CACHE)
UNSET(MMU_ENABLE_INTERNAL_TESTS CACHE)
UNSET(MMU_ENABLE_ARCH_FLAGS CACHE)
UNSET(MMU_ENABLE_ARCH_FLAGS_SIMPLE CACHE)
