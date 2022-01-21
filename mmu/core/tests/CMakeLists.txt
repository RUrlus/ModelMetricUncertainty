CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

################################################################################
#                               PYTHON EXTENSION                               #
################################################################################
SET(MODNAME "_mmu_core_tests")
SET(MMU_EXTENSION_TEST_PATH "${PROJECT_SOURCE_DIR}/mmu/core/tests")


pybind11_add_module(${MODNAME}
    MODULE
        ${MMU_EXTENSION_TEST_PATH}/bindings.cpp
)

TARGET_LINK_LIBRARIES(${MODNAME} PUBLIC mmu::headers)
TARGET_INCLUDE_DIRECTORIES(${MODNAME} PUBLIC ${MMU_EXTENSION_TEST_PATH})
# Set required C++14 flag
SET_PROPERTY(TARGET ${MODNAME} PROPERTY CXX_STANDARD 14)
SET_PROPERTY(TARGET ${MODNAME} PROPERTY CXX_STANDARD_REQUIRED ON)
SET_PROPERTY(TARGET ${MODNAME} PROPERTY CXX_EXTENSIONS OFF)
SET_PROPERTY(TARGET ${MODNAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

IF(CALL_FROM_SETUP_PY)
  # cmake-build-extension sets the full absolute path as CMAKE_INSTALL_PREFIX.
  SET(MMU_CORE_TEST_INSTDIR "${CMAKE_INSTALL_PREFIX}/mmu/lib")
ELSE()
  SET(MMU_CORE_TEST_INSTDIR "${PROJECT_SOURCE_DIR}/mmu/lib")
ENDIF()

MESSAGE(STATUS "MMU: Installing ${MODNAME} in ${MMU_CORE_TEST_INSTDIR}")

# Setup installation path
INSTALL(TARGETS ${MODNAME} DESTINATION "${MMU_CORE_TEST_INSTDIR}")
