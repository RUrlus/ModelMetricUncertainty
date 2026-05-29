# Header-only target used by tests and extension.
add_library(mmu_headers INTERFACE)
add_library(mmu::headers ALIAS mmu_headers)
target_include_directories(mmu_headers INTERFACE ${PROJECT_SOURCE_DIR}/src/mmu_core/include/)

# Full interface target with optional OpenMP and warning flags.
add_library(mmu_mmu INTERFACE)
add_library(mmu::mmu ALIAS mmu_mmu)
target_include_directories(
  mmu_mmu INTERFACE
    ${PROJECT_SOURCE_DIR}/src/mmu_core/include/
    ${Python3_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src/mmu_core/external/pcg-cpp/include
)

if(MMU_DEV_MODE AND NOT MMU_CICD_MODE)
  target_compile_options(mmu_mmu INTERFACE ${MMU_DEVMODE_OPTIONS})
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
    ${MMU_API_PATH}/ppn_recall_multn_loglike.cpp
    ${MMU_BINDINGS_PATH}/ppn_recall_multn_loglike.cpp)

pybind11_add_module(_mmu_core MODULE ${MMU_SRC_FILES})
target_link_libraries(_mmu_core PUBLIC mmu::mmu)
target_compile_definitions(
  _mmu_core PRIVATE EXTENSION_MODULE_NAME=_mmu_core VERSION_INFO=${PROJECT_VERSION}
)

if(MMU_ARCHITECTURE_FLAGS)
  target_compile_options(_mmu_core PRIVATE
    "$<$<CONFIG:Release>:${MMU_ARCHITECTURE_FLAGS}>"
  )
endif()

if(MMU_X86_ISA_NAME)
  target_compile_definitions(_mmu_core PRIVATE MMU_X86_ISA_${MMU_X86_ISA_NAME}=1)
endif()

set_property(TARGET _mmu_core PROPERTY CXX_STANDARD ${MMU_CPP_STANDARD})
set_property(TARGET _mmu_core PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET _mmu_core PROPERTY POSITION_INDEPENDENT_CODE ON)

if(OpenMP_CXX_FOUND AND APPLE)
  include(SetHomebrew)
  # Keep the extension loadable on macOS by rewriting the linked OpenMP path.
  get_target_property(
    OpenMP_LIBRARY_LOCATION
    OpenMP::OpenMP_CXX
    INTERFACE_LINK_LIBRARIES
  )
  get_filename_component(OpenMP_LIBRARY_NAME ${OpenMP_LIBRARY_LOCATION} NAME)
  get_filename_component(OpenMP_LIBRARY_DIR ${OpenMP_LIBRARY_LOCATION} DIRECTORY)

  add_custom_command(
    TARGET _mmu_core
    POST_BUILD
    COMMAND
      install_name_tool
      -change
      ${OpenMP_LIBRARY_LOCATION}
      "@rpath/${OpenMP_LIBRARY_NAME}"
      $<TARGET_FILE:_mmu_core>
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Replacing hard-coded OpenMP install_name with @rpath/${OpenMP_LIBRARY_NAME}"
  )

  if(MMU_VENDOR_OPENMP)
    set(OpenMP_TARGET_DIR "${CMAKE_INSTALL_PREFIX}/${PROJECT_NAME}/.dylibs")
    set(OpenMP_TARGET_LOCATION "${OpenMP_TARGET_DIR}/${OpenMP_LIBRARY_NAME}")
    message(STATUS "mmu: Copying ${OpenMP_LIBRARY_NAME} to ${OpenMP_TARGET_LOCATION}")
    add_custom_command(
      TARGET _mmu_core
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory ${OpenMP_TARGET_DIR}
      COMMAND ${CMAKE_COMMAND} -E copy ${OpenMP_LIBRARY_LOCATION} ${OpenMP_TARGET_LOCATION}
    )
    set_target_properties(
      _mmu_core
      PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "@loader_path/../.dylibs/"
        INSTALL_RPATH_USE_LINK_PATH FALSE
    )
  else()
    set_target_properties(
      _mmu_core
      PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH
          "${OpenMP_LIBRARY_DIR};${MMU_HOMEBREW_PREFIX}/opt/libomp/lib;/opt/homebrew/opt/libomp/lib"
        INSTALL_RPATH_USE_LINK_PATH FALSE
    )
  endif()
endif()

if(SKBUILD)
  install(TARGETS _mmu_core LIBRARY DESTINATION "${PROJECT_NAME}/lib")
else()
  install(TARGETS _mmu_core LIBRARY DESTINATION "${PROJECT_SOURCE_DIR}/src/mmu/lib")
endif()
