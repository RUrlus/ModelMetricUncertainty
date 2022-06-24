IF (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  MESSAGE(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  SET(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "Asan" "MinSizeRel" "RelWithDebInfo")
ENDIF ()

IF (MMU_VALGRIND_MODE)
    MESSAGE(STATUS "mmu: Valgrind mode selected")
    STRING(REGEX REPLACE "-DNDEBUG " "" CMAKE_CXX_FLAGS_RELWITHDEBUG "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DDEBUG" )
    STRING(REGEX REPLACE "-DNDEBUG " "" CMAKE_C_FLAGS_RELWITHDEBUG "${CMAKE_C_FLAGS_RELWITHDEBINFO} -DDEBUG" )
    SET(MMU_DISABLE_OPENMP ON)
    SET(MMU_ENABLE_ARCH_FLAGS OFF)
    SET(MMU_MMU_CICD_MODE ON)
ENDIF ()

IF (MMU_CICD_MODE)
    SET(MMU_ARCHITECTURE_FLAGS "")
    MESSAGE(STATUS "mmu: CICD Mode")
ELSEIF (MMU_ENABLE_ARCH_FLAGS)
    MESSAGE(STATUS "mmu: Building for native host")
    INCLUDE(OptimizeForArchitecture)
    OptimizeForArchitecture()
ELSEIF (MMU_ENABLE_ARCH_FLAGS_SIMPLE)
    MESSAGE(STATUS "mmu: Building for native host")
    INCLUDE(SimpleOptimizeForArchitecture)
ELSE ()
    SET(MMU_ARCHITECTURE_FLAGS "")
    MESSAGE(STATUS "mmu: Building for non-native host")
ENDIF()

IF (MMU_COVERAGE)
    # --coverage option is used to compile and link code instrumented for coverage analysis.
    # The option is a synonym for
    #    -fprofile-arcs
    #    -ftest-coverage (when compiling)
    #    -lgcov (when linking).
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
ENDIF ()

IF (CMAKE_BUILD_TYPE EQUAL "Asan")
    SET(CMAKE_C_FLAGS_ASAN
        "${CMAKE_C_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer" CACHE STRING
        "Flags used by the C compiler for Asan build type or configuration." FORCE)
    
    SET(CMAKE_CXX_FLAGS_ASAN
        "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer" CACHE STRING
        "Flags used by the C++ compiler for Asan build type or configuration." FORCE)
    
    SET(CMAKE_EXE_LINKER_FLAGS_ASAN
        "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -fsanitize=address" CACHE STRING
        "Linker flags to be used to create executables for Asan build type." FORCE)
    
    SET(CMAKE_SHARED_LINKER_FLAGS_ASAN
        "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -fsanitize=address" CACHE STRING
        "Linker lags to be used to create shared libraries for Asan build type." FORCE)
ENDIF ()
