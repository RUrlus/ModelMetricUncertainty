# Check for AVX/AVX2 and set AVX_FLAGS for supported toolchains.
function(MMU_CHECK_FOR_AVX)
  set(AVX_FLAGS)
  include(CheckCXXSourceRuns)

  if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    set(CMAKE_REQUIRED_FLAGS "-march=native -mtune=native")
  endif()

  check_cxx_source_runs(
    "#include <immintrin.h>
     int main() {
       __m256i a = {0};
       a = _mm256_abs_epi16(a);
       return 0;
     }"
    HAVE_AVX2)

  if(NOT HAVE_AVX2)
    check_cxx_source_runs(
      "#include <immintrin.h>
       int main() {
         __m256 a;
         a = _mm256_set1_ps(0);
         return 0;
       }"
      HAVE_AVX)
  endif()

  set(CMAKE_REQUIRED_FLAGS)

  if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    if(HAVE_AVX2)
      set(AVX_FLAGS "-mavx2" PARENT_SCOPE)
    elseif(HAVE_AVX)
      set(AVX_FLAGS "-mavx" PARENT_SCOPE)
    endif()
  endif()

  if(MSVC)
    if(HAVE_AVX2)
      set(AVX_FLAGS "/arch:AVX2" PARENT_SCOPE)
    elseif(HAVE_AVX)
      set(AVX_FLAGS "/arch:AVX" PARENT_SCOPE)
    endif()
  endif()
endfunction()

