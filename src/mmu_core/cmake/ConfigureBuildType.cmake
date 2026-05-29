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
    PROPERTY STRINGS "Debug" "Release" "Asan" "MinSizeRel" "RelWithDebInfo"
  )
endif()

if(MMU_VALGRIND_MODE)
  message(STATUS "mmu: Valgrind mode selected")
  string(
    REGEX REPLACE "-DNDEBUG " "" CMAKE_CXX_FLAGS_RELWITHDEBUG
                  "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DDEBUG"
  )
  string(
    REGEX REPLACE "-DNDEBUG " "" CMAKE_C_FLAGS_RELWITHDEBUG
                  "${CMAKE_C_FLAGS_RELWITHDEBINFO} -DDEBUG"
  )
  set(MMU_DISABLE_OPENMP ON)
  set(MMU_ENABLE_ARCH_FLAGS OFF)
  set(MMU_CICD_MODE ON)
endif()

include(CheckCXXCompilerFlag)

function(mmu_check_cxx_support FLAG DEST)
  string(REGEX REPLACE "^[-/]" "" _MMU_FLAG_NAME "${FLAG}")
  string(REGEX REPLACE "[^A-Za-z0-9_]" "_" _MMU_FLAG_NAME "${_MMU_FLAG_NAME}")
  string(TOUPPER "${_MMU_FLAG_NAME}" _MMU_FLAG_NAME)
  set(_MMU_RES_VAR "${_MMU_FLAG_NAME}_SUPPORTED")

  check_cxx_compiler_flag("${FLAG}" ${_MMU_RES_VAR})
  if(${_MMU_RES_VAR})
    set(${DEST} ${${DEST}} "${FLAG}" PARENT_SCOPE)
  endif()
endfunction()

# MMU_ARCHITECTURE_FLAGS is always treated as a CMake list.
set(MMU_ARCHITECTURE_FLAGS)
unset(MMU_X86_ISA_NAME)

if(MMU_CICD_MODE)
  message(STATUS "mmu: CICD Mode")

elseif(MMU_ENABLE_ARCH_FLAGS)
  message(STATUS "mmu: Building with host ISA auto-detection")

  # Normalize processor string for matching.
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" MMU_SYSTEM_PROCESSOR_LOWER)

  # ARM / AArch64: use native host tuning only.
  if(MMU_SYSTEM_PROCESSOR_LOWER MATCHES "^(arm64|aarch64)$")
    mmu_check_cxx_support("-march=native" MMU_ARCHITECTURE_FLAGS)

    if(MMU_ARCHITECTURE_FLAGS)
      string(JOIN " " MMU_ARCHITECTURE_FLAGS_STR ${MMU_ARCHITECTURE_FLAGS})
      message(STATUS "mmu: Enabled ARM host ISA flags: ${MMU_ARCHITECTURE_FLAGS_STR}")
    else()
      message(STATUS "mmu: ARM host ISA auto-detection enabled but no supported flags were validated")
    endif()

  # x86 / x86_64: detect highest usable ISA tier.
  elseif(MMU_SYSTEM_PROCESSOR_LOWER MATCHES "^(x86_64|amd64|x86|i[3-6]86)$")
    include(FindX86Isa)
    MMU_DETECT_X86_ISA()

    if(MMU_X86_ISA_FLAGS)
      set(MMU_ARCHITECTURE_FLAGS ${MMU_X86_ISA_FLAGS})
      string(JOIN " " MMU_ARCHITECTURE_FLAGS_STR ${MMU_ARCHITECTURE_FLAGS})
      message(STATUS "mmu: Enabled x86 host ISA tier: ${MMU_X86_ISA_NAME}")
      message(STATUS "mmu: Enabled x86 host ISA flags: ${MMU_ARCHITECTURE_FLAGS_STR}")
    elseif(MMU_X86_ISA_NAME)
      # Possible on MSVC x64 for SSE/SSE4.2 where no explicit /arch flag is needed.
      message(STATUS "mmu: Enabled x86 host ISA tier: ${MMU_X86_ISA_NAME} (no explicit compiler flags required)")
    else()
      message(STATUS "mmu: x86 host ISA auto-detection enabled but no usable ISA tier was validated")
    endif()

  else()
    message(STATUS "mmu: Host ISA auto-detection enabled, but no architecture-specific policy exists for processor '${CMAKE_SYSTEM_PROCESSOR}'")
  endif()

else()
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
