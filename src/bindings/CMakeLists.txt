CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

################################################################################
#                               PYTHON EXTENSION                               #
################################################################################
SET(MODNAME "_mmu_core")
ADD_DEFINITIONS(-DEXTENSION_MODULE_NAME=${MODNAME})

SET(MMU_BINDINGS_PATH "${PROJECT_SOURCE_DIR}/src/bindings")
SET(MMU_API_PATH "${PROJECT_SOURCE_DIR}/src/api")

pybind11_add_module(${MODNAME}
    MODULE
        ${MMU_BINDINGS_PATH}/bindings.cpp
        ${MMU_BINDINGS_PATH}/utils.cpp
        ${MMU_BINDINGS_PATH}/confusion_matrix.cpp
        ${MMU_API_PATH}/metrics.cpp
        ${MMU_BINDINGS_PATH}/metrics.cpp
        ${MMU_API_PATH}/bvn_error.cpp
        ${MMU_BINDINGS_PATH}/bvn_error.cpp
        ${MMU_API_PATH}/pr_bvn_grid.cpp
        ${MMU_BINDINGS_PATH}/pr_bvn_grid.cpp
        ${MMU_API_PATH}/roc_bvn_grid.cpp
        ${MMU_BINDINGS_PATH}/roc_bvn_grid.cpp
        ${MMU_API_PATH}/pr_multn_loglike.cpp
        ${MMU_BINDINGS_PATH}/pr_multn_loglike.cpp
        ${MMU_API_PATH}/roc_multn_loglike.cpp
        ${MMU_BINDINGS_PATH}/roc_multn_loglike.cpp
)

TARGET_LINK_LIBRARIES(${MODNAME} PUBLIC mmu::mmu)
TARGET_COMPILE_DEFINITIONS(${MODNAME} PRIVATE VERSION_INFO=${MMU_VERSION_INFO})
TARGET_COMPILE_OPTIONS(${MODNAME} PRIVATE "$<$<CONFIG:RELEASE>:${MMU_ARCHITECTURE_FLAGS}>")

# Set required C++14 flags
SET_PROPERTY(TARGET ${MODNAME} PROPERTY CXX_STANDARD 14)
SET_PROPERTY(TARGET ${MODNAME} PROPERTY CXX_STANDARD_REQUIRED ON)
SET_PROPERTY(TARGET ${MODNAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

if(SKBUILD)
    INSTALL(TARGETS ${MODNAME} LIBRARY DESTINATION "${PROJECT_NAME}/lib")
ELSE ()
    INSTALL(TARGETS ${MODNAME} LIBRARY DESTINATION "${PROJECT_SOURCE_DIR}/mmu/lib")
ENDIF ()
