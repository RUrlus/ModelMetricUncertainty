CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

IF (NOT DEFINED MMU_CORE_VERSION_INFO)
    SET(MMU_CORE_VERSION_INFO "0.1.0")
ENDIF ()

PROJECT(
    mmu_core_tests
    VERSION ${MMU_CORE_VERSION_INFO}
    DESCRIPTION "Fast binary classification metrics."
    LANGUAGES CXX
)

SET(CMAKE_MODULE_PATH_SAVED ${CMAKE_MODULE_PATH})
LIST(INSERT CMAKE_MODULE_PATH 0 "${PROJECT_SOURCE_DIR}/../cmake")
################################################################################
#                                   SETTINGS                                   #
################################################################################
SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

INCLUDE(MMU_ReleaseFlags)

FIND_PACKAGE(Python3 COMPONENTS Development NumPy REQUIRED)
# Needed to ensure the same Python executable is found by Pybind11
IF (NOT DEFINED PYTHON_EXECUTABLE)
    SET(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
ENDIF ()

FIND_PACKAGE(pybind11 CONFIG REQUIRED)
FIND_PACKAGE(OpenMP)

################################################################################
#                                HEADER TARGET                                 #
################################################################################
SET(MMU_HEADERS ${PROJECT_SOURCE_DIR}/../cpp/include)
INCLUDE(MMU_headers)

SET(MODNAME "_mmu_core_tests")
SET(PROJ_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/mmu)

pybind11_add_module(${MODNAME}
    MODULE
        ${PROJECT_SOURCE_DIR}/cpp/bindings.cpp
        ${PROJECT_SOURCE_DIR}/cpp/numpy_bindings.hpp
)

TARGET_LINK_LIBRARIES(${MODNAME} PUBLIC mmu::headers)
TARGET_INCLUDE_DIRECTORIES(${MODNAME} PUBLIC ${PROJECT_SOURCE_DIR}/cpp/tests)
# Set required C++14 flag
SET_PROPERTY(TARGET ${MODNAME} PROPERTY CXX_STANDARD 14)
SET_PROPERTY(TARGET ${MODNAME} PROPERTY CXX_STANDARD_REQUIRED ON)
SET_PROPERTY(TARGET ${MODNAME} PROPERTY CXX_EXTENSIONS OFF)
SET_PROPERTY(TARGET ${MODNAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

SET(MMU_CORE_INSTDIR "${PROJECT_SOURCE_DIR}/mmu_tests/lib")
MESSAGE(STATUS "Installing _mmu_core_tests in ${MMU_CORE_INSTDIR}")
# Setup installation path
INSTALL(TARGETS ${MODNAME} DESTINATION "${MMU_CORE_INSTDIR}")
