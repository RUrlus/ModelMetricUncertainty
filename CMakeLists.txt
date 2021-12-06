CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

IF (NOT DEFINED MMU_CORE_VERSION_INFO)
    SET(MMU_CORE_VERSION_INFO "0.1.0")
ENDIF ()

PROJECT(
    mmu_core
    VERSION ${MMU_CORE_VERSION_INFO}
    DESCRIPTION "Fast binary classification metrics."
    LANGUAGES CXX
)

SET(CMAKE_MODULE_PATH_SAVED ${CMAKE_MODULE_PATH})
LIST(INSERT CMAKE_MODULE_PATH 0 "${PROJECT_SOURCE_DIR}/cmake")
################################################################################
#                                   SETTINGS                                   #
################################################################################
SET(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

INCLUDE(MMU_ReleaseFlags)

FIND_PACKAGE(Python3 COMPONENTS Development NumPy REQUIRED)
# Needed to ensure the same Python executable is found by Pybind11
IF (NOT DEFINED PYTHON_EXECUTABLE)
    SET(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
ENDIF ()

FIND_PACKAGE(pybind11 CONFIG REQUIRED)
FIND_PACKAGE(OpenMP)

ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/cpp)
