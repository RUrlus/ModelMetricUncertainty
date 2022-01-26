CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

IF (NOT DEFINED MMU_VERSION_INFO)
    SET(MMU_VERSION_INFO "0.1.0")
ENDIF ()

SET(DEFAULT_BUILD_TYPE Release)
IF (DEFINED ENV{MMU_DEBUG_BUILD})
    SET(CMAKE_BUILD_TYPE Debug)
ENDIF ()
IF (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  MESSAGE(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  SET(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
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
ADD_LIBRARY(mmu_headers INTERFACE)
ADD_LIBRARY(mmu::headers ALIAS mmu_headers)
TARGET_INCLUDE_DIRECTORIES(mmu_headers 
    INTERFACE
        ${PROJECT_SOURCE_DIR}/mmu/core/include
        ${Python3_INCLUDE_DIRS}
)

TARGET_LINK_LIBRARIES(
    mmu_headers
    INTERFACE
        pybind11::pybind11
        Python3::NumPy
)

IF (OpenMP_CXX_FOUND)
  TARGET_LINK_LIBRARIES(
      mmu_headers
      INTERFACE
          OpenMP::OpenMP_CXX
  )
ENDIF ()

ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/mmu/core)
