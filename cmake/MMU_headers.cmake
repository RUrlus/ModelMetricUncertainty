CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

################################################################################
#                                HEADER TARGET                                 #
################################################################################
ADD_LIBRARY(mmu_headers INTERFACE)
ADD_LIBRARY(mmu::headers ALIAS mmu_headers)
TARGET_INCLUDE_DIRECTORIES(mmu_headers INTERFACE ${MMU_HEADERS})
TARGET_LINK_LIBRARIES(
    mmu_headers
    INTERFACE
        pybind11::pybind11
        OpenMP::OpenMP_CXX
        Python3::NumPy
        Python3::Python
)