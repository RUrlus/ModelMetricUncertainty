CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

################################################################################
#                                HEADER TARGET                                 #
################################################################################
SET(MMU_HEADERS ${PROJECT_SOURCE_DIR}/cpp/include)
INCLUDE(MMU_headers)

################################################################################
#                               PYTHON EXTENSION                               #
################################################################################
SET(MODNAME "_mmu_core")
ADD_DEFINITIONS(-DEXTENSION_MODULE_NAME=${MODNAME})
SET(PROJ_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/mmu)

pybind11_add_module(${MODNAME}
    MODULE
        ${PROJECT_SOURCE_DIR}/cpp/bindings/bindings.cpp
        ${PROJECT_SOURCE_DIR}/cpp/bindings/metrics.hpp
        ${PROJECT_SOURCE_DIR}/cpp/bindings/metrics.cpp
        ${PROJECT_SOURCE_DIR}/cpp/bindings/confusion_matrix.hpp
        ${PROJECT_SOURCE_DIR}/cpp/bindings/confusion_matrix.cpp
)

TARGET_LINK_LIBRARIES(${MODNAME} PUBLIC mmu::headers)
TARGET_INCLUDE_DIRECTORIES(${MODNAME} PUBLIC ${PROJECT_SOURCE_DIR}/cpp/bindings)
TARGET_COMPILE_DEFINITIONS(${MODNAME} PRIVATE VERSION_INFO=${MMU_VERSION_INFO})
TARGET_COMPILE_OPTIONS(${MODNAME} PRIVATE "$<$<CONFIG:RELEASE>:${MMU_RELEASE_OPTIONS}>")

# Set required C++14 flags
SET_PROPERTY(TARGET ${MODNAME} PROPERTY CXX_STANDARD 14)
SET_PROPERTY(TARGET ${MODNAME} PROPERTY CXX_STANDARD_REQUIRED ON)
SET_PROPERTY(TARGET ${MODNAME} PROPERTY CXX_EXTENSIONS OFF)
SET_PROPERTY(TARGET ${MODNAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

IF(CALL_FROM_SETUP_PY)
  # cmake-build-extension sets the full absolute path as CMAKE_INSTALL_PREFIX.
  SET(MMU_CORE_INSTDIR "${CMAKE_INSTALL_PREFIX}")
ELSE()
  SET(MMU_CORE_INSTDIR "${PROJECT_SOURCE_DIR}/mmu/lib")
ENDIF()

MESSAGE(STATUS "Installing _mmu_core in ${MMU_CORE_INSTDIR}")

# Setup installation path
INSTALL(TARGETS ${MODNAME} DESTINATION "${MMU_CORE_INSTDIR}")
