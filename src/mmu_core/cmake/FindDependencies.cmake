include(GNUInstallDirs)

find_package(
  Python3
  COMPONENTS Development NumPy
  QUIET)
if(NOT Python3_FOUND)
  find_package(
    Python3
    COMPONENTS Development.Module NumPy
    REQUIRED)
endif()

# Ensure pybind11 discovers the same interpreter.
if(NOT DEFINED PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
endif()

find_package(pybind11 CONFIG REQUIRED)
if(MMU_ENABLE_OPENMP)
  find_package(OpenMP)
  if((NOT OpenMP_FOUND) AND APPLE)
    include(SetHomebrew)
    set(OpenMP_ROOT "${MMU_HOMEBREW_PREFIX}/opt/libomp")
    find_package(OpenMP)
  endif()
  find_package(OpenMP REQUIRED)
elseif(MMU_DISABLE_OPENMP)
  message(STATUS "mmu: OpenMP is disabled")
else()
  find_package(OpenMP)
  if((NOT OpenMP_FOUND) AND APPLE)
    include(SetHomebrew)
    set(OpenMP_ROOT "${MMU_HOMEBREW_PREFIX}/opt/libomp")
    find_package(OpenMP)
  endif()
endif()

