# Check for SSE support and set SSE_FLAGS/SSE_DEFINITIONS.
function(MMU_CHECK_FOR_SSE)
  set(SSE_FLAGS)
  set(SSE_DEFINITIONS)

  include(CheckCXXSourceRuns)
  set(CMAKE_REQUIRED_FLAGS)
  set(SSE_LEVEL 0)

  if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    set(CMAKE_REQUIRED_FLAGS "-msse4.2")
  endif()

  check_cxx_source_runs(
    "#include <emmintrin.h>
     #include <nmmintrin.h>
     int main() {
       long long a[2] = {1, 2};
       long long b[2] = {-1, 3};
       long long c[2];
       __m128i va = _mm_loadu_si128((__m128i*)a);
       __m128i vb = _mm_loadu_si128((__m128i*)b);
       __m128i vc = _mm_cmpgt_epi64(va, vb);
       _mm_storeu_si128((__m128i*)c, vc);
       return (c[0] == -1LL && c[1] == 0LL) ? 0 : 1;
     }"
    HAVE_SSE4_2_EXTENSIONS)

  if(HAVE_SSE4_2_EXTENSIONS)
    set(SSE_LEVEL 4.2)
  endif()

  if(SSE_LEVEL LESS 4.2)
    if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
      set(CMAKE_REQUIRED_FLAGS "-msse2")
    elseif(MSVC AND NOT CMAKE_CL_64)
      set(CMAKE_REQUIRED_FLAGS "/arch:SSE2")
    endif()

    check_cxx_source_runs(
      "#include <emmintrin.h>
       int main() {
         __m128d a, b;
         double vals[2] = {0};
         a = _mm_loadu_pd(vals);
         b = _mm_add_pd(a, a);
         _mm_storeu_pd(vals, b);
         return 0;
       }"
      HAVE_SSE2_EXTENSIONS)

    if(HAVE_SSE2_EXTENSIONS)
      set(SSE_LEVEL 2.0)
    endif()
  endif()

  if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    if(SSE_LEVEL GREATER_EQUAL 4.2)
      set(SSE_FLAGS "-msse4.2 -mfpmath=sse")
    elseif(SSE_LEVEL GREATER_EQUAL 2.0)
      set(SSE_FLAGS "-msse2 -mfpmath=sse")
    endif()
  elseif(MSVC AND NOT CMAKE_CL_64)
    if(SSE_LEVEL GREATER_EQUAL 2.0)
      set(SSE_FLAGS "/arch:SSE2")
    endif()
  endif()

  set(SSE_FLAGS ${SSE_FLAGS} PARENT_SCOPE)
  set(SSE_DEFINITIONS ${SSE_DEFINITIONS} PARENT_SCOPE)
  set(CMAKE_REQUIRED_FLAGS)
endfunction()

