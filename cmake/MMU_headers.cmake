CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

################################################################################
#                                HEADER TARGET                                 #
################################################################################
ADD_LIBRARY(mmu_headers INTERFACE)
ADD_LIBRARY(mmu::headers ALIAS mmu_headers)
TARGET_INCLUDE_DIRECTORIES(mmu_headers 
    INTERFACE
        ${MMU_HEADERS}
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
