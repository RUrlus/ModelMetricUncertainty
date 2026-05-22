if(MMU_ENABLE_DEVMODE)
  set(MMU_DEV_MODE ON)
endif()

set(CMAKE_CXX_STANDARD ${MMU_CPP_STANDARD})
if((CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
   OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
   OR (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"))
  set(MMU_DEVMODE_OPTIONS -Wall -Wextra -Wunused-variable -Wunused-const-variable)
else()
  set(MMU_DEVMODE_OPTIONS)
endif()

if(DEFINED MMU_ENABLE_DEBUG AND MMU_ENABLE_DEBUG)
  set(MMU_DEFAULT_BUILD_TYPE Debug)
else()
  set(MMU_DEFAULT_BUILD_TYPE Release)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${MMU_DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${MMU_DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
  set_property(
    CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS "Debug" "Release" "Asan" "MinSizeRel" "RelWithDebInfo")
endif()

if(MMU_VALGRIND_MODE)
  message(STATUS "mmu: Valgrind mode selected")
  string(
    REGEX REPLACE "-DNDEBUG " "" CMAKE_CXX_FLAGS_RELWITHDEBUG
                  "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DDEBUG")
  string(
    REGEX REPLACE "-DNDEBUG " "" CMAKE_C_FLAGS_RELWITHDEBUG
                  "${CMAKE_C_FLAGS_RELWITHDEBINFO} -DDEBUG")
  set(MMU_DISABLE_OPENMP ON)
  set(MMU_ENABLE_ARCH_FLAGS OFF)
  set(MMU_CICD_MODE ON)
endif()

if(MMU_CICD_MODE)
  set(MMU_ARCHITECTURE_FLAGS "")
  message(STATUS "mmu: CICD Mode")
elseif(MMU_ENABLE_ARCH_FLAGS)
  message(STATUS "mmu: Building with architecture flag auto-detection")
  set(MMU_ARCHITECTURE_FLAGS "")

  include(CheckCXXCompilerFlag)
  function(mmu_check_cxx_support FLAG DEST)
    string(SUBSTRING ${FLAG} 1 -1 STRIPPED_FLAG)
    string(REGEX REPLACE "=" "_" STRIPPED_FLAG ${STRIPPED_FLAG})
    string(TOUPPER ${STRIPPED_FLAG} STRIPPED_FLAG)
    set(RES_VAR "${STRIPPED_FLAG}_SUPPORTED")
    check_cxx_compiler_flag("${FLAG}" ${RES_VAR})
    if(${RES_VAR})
      set(${DEST} "${${DEST}} ${FLAG}" PARENT_SCOPE)
    endif()
  endfunction()

  if(APPLE AND (CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64"))
    mmu_check_cxx_support("-march=native" MMU_ARCHITECTURE_FLAGS)
  else()
    include(FindSse)
    include(FindAvx)
    MMU_CHECK_FOR_SSE()
    MMU_CHECK_FOR_AVX()
    string(APPEND MMU_ARCHITECTURE_FLAGS " ${SSE_FLAGS} ${AVX_FLAGS}")
  endif()

  string(STRIP "${MMU_ARCHITECTURE_FLAGS}" MMU_ARCHITECTURE_FLAGS)
  if(NOT "${MMU_ARCHITECTURE_FLAGS}" STREQUAL "")
    message(STATUS "mmu: Enabled arch flags: ${MMU_ARCHITECTURE_FLAGS}")
  else()
    message(STATUS "mmu: Architecture flags enabled but none were validated")
  endif()
else()
  set(MMU_ARCHITECTURE_FLAGS "")
  message(STATUS "mmu: Building for non-native host")
endif()

if(MMU_COVERAGE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Asan")
  set(CMAKE_C_FLAGS_ASAN
      "${CMAKE_C_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer"
      CACHE STRING "Flags used by the C compiler for Asan build type or configuration."
      FORCE)
  set(CMAKE_CXX_FLAGS_ASAN
      "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer"
      CACHE STRING "Flags used by the C++ compiler for Asan build type or configuration."
      FORCE)
  set(CMAKE_EXE_LINKER_FLAGS_ASAN
      "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -fsanitize=address"
      CACHE STRING "Linker flags to be used to create executables for Asan build type."
      FORCE)
  set(CMAKE_SHARED_LINKER_FLAGS_ASAN
      "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -fsanitize=address"
      CACHE STRING "Linker flags to be used to create shared libraries for Asan build type."
      FORCE)
endif()

if(MMU_DEV_MODE)
  set(MMU_ENABLE_INTERNAL_TESTS ON)
endif()

