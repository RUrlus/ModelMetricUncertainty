# Detect the highest usable x86 ISA tier and return:
#   MMU_X86_ISA_FLAGS : list of compiler flags for the selected tier
#   MMU_X86_ISA_NAME  : one of AVX2, AVX, SSE42, SSE2
#
# Policy:
# - select a single highest supported tier
# - do not accumulate overlapping lower-tier flags
# - intended for native-host tuning, so uses check_cxx_source_runs()
#
# Supported toolchains:
# - GNU / Clang / AppleClang
# - MSVC
#
# Supported architectures:
# - x86 / x86_64 / AMD64 only

include(CheckCXXSourceRuns)

function(_mmu_check_x86_isa_runs FLAG SOURCE RESULT_VAR)
  set(_MMU_OLD_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
  set(CMAKE_REQUIRED_FLAGS "${FLAG}")

  check_cxx_source_runs("${SOURCE}" ${RESULT_VAR})

  set(CMAKE_REQUIRED_FLAGS "${_MMU_OLD_REQUIRED_FLAGS}")
endfunction()

function(MMU_DETECT_X86_ISA)
  set(MMU_X86_ISA_FLAGS)
  set(MMU_X86_ISA_NAME)

  # AVX2 probe
  if(MSVC)
    set(_MMU_AVX2_FLAG "/arch:AVX2")
  else()
    set(_MMU_AVX2_FLAG "-mavx2")
  endif()

  _mmu_check_x86_isa_runs(
    "${_MMU_AVX2_FLAG}"
    "#include <immintrin.h>
     int main() {
       __m256i a = _mm256_set1_epi16(1);
       __m256i b = _mm256_abs_epi16(a);
       return _mm256_extract_epi16(b, 0) == 1 ? 0 : 1;
     }"
    MMU_HAVE_X86_AVX2
  )

  if(MMU_HAVE_X86_AVX2)
    if(MSVC)
      set(MMU_X86_ISA_FLAGS /arch:AVX2)
    else()
      set(MMU_X86_ISA_FLAGS -mavx2)
    endif()
    set(MMU_X86_ISA_NAME AVX2)

    set(MMU_X86_ISA_FLAGS ${MMU_X86_ISA_FLAGS} PARENT_SCOPE)
    set(MMU_X86_ISA_NAME ${MMU_X86_ISA_NAME} PARENT_SCOPE)
    return()
  endif()

  # AVX probe
  if(MSVC)
    set(_MMU_AVX_FLAG "/arch:AVX")
  else()
    set(_MMU_AVX_FLAG "-mavx")
  endif()

  _mmu_check_x86_isa_runs(
    "${_MMU_AVX_FLAG}"
    "#include <immintrin.h>
     int main() {
       __m256 a = _mm256_set1_ps(1.0f);
       __m256 b = _mm256_add_ps(a, a);
       float out[8];
       _mm256_storeu_ps(out, b);
       return out[0] == 2.0f ? 0 : 1;
     }"
    MMU_HAVE_X86_AVX
  )

  if(MMU_HAVE_X86_AVX)
    if(MSVC)
      set(MMU_X86_ISA_FLAGS /arch:AVX)
    else()
      set(MMU_X86_ISA_FLAGS -mavx)
    endif()
    set(MMU_X86_ISA_NAME AVX)

    set(MMU_X86_ISA_FLAGS ${MMU_X86_ISA_FLAGS} PARENT_SCOPE)
    set(MMU_X86_ISA_NAME ${MMU_X86_ISA_NAME} PARENT_SCOPE)
    return()
  endif()

  # SSE4.2 probe
  if(MSVC)
    # MSVC x64 already implies SSE2 baseline; no separate SSE4.2 /arch switch exists.
    # Still probe functionality without an explicit flag.
    set(_MMU_SSE42_FLAG "")
  else()
    set(_MMU_SSE42_FLAG "-msse4.2")
  endif()

  _mmu_check_x86_isa_runs(
    "${_MMU_SSE42_FLAG}"
    "#include <emmintrin.h>
     #include <nmmintrin.h>
     int main() {
       long long a[2] = {1, 2};
       long long b[2] = {-1, 3};
       long long c[2];
       __m128i va = _mm_loadu_si128((const __m128i*)a);
       __m128i vb = _mm_loadu_si128((const __m128i*)b);
       __m128i vc = _mm_cmpgt_epi64(va, vb);
       _mm_storeu_si128((__m128i*)c, vc);
       return (c[0] == -1LL && c[1] == 0LL) ? 0 : 1;
     }"
    MMU_HAVE_X86_SSE42
  )

  if(MMU_HAVE_X86_SSE42)
    if(MSVC)
      # No dedicated /arch:SSE4.2. Leave flags empty on MSVC.
      set(MMU_X86_ISA_FLAGS)
    else()
      set(MMU_X86_ISA_FLAGS -msse4.2)
    endif()
    set(MMU_X86_ISA_NAME SSE42)

    set(MMU_X86_ISA_FLAGS ${MMU_X86_ISA_FLAGS} PARENT_SCOPE)
    set(MMU_X86_ISA_NAME ${MMU_X86_ISA_NAME} PARENT_SCOPE)
    return()
  endif()

  # SSE2 probe
  if(MSVC)
    if(CMAKE_CL_64)
      set(_MMU_SSE2_FLAG "")
    else()
      set(_MMU_SSE2_FLAG "/arch:SSE2")
    endif()
  else()
    set(_MMU_SSE2_FLAG "-msse2")
  endif()

  _mmu_check_x86_isa_runs(
    "${_MMU_SSE2_FLAG}"
    "#include <emmintrin.h>
     int main() {
       __m128d a = _mm_set1_pd(1.0);
       __m128d b = _mm_add_pd(a, a);
       double out[2];
       _mm_storeu_pd(out, b);
       return out[0] == 2.0 ? 0 : 1;
     }"
    MMU_HAVE_X86_SSE2
  )

  if(MMU_HAVE_X86_SSE2)
    if(MSVC)
      if(CMAKE_CL_64)
        set(MMU_X86_ISA_FLAGS)
      else()
        set(MMU_X86_ISA_FLAGS /arch:SSE2)
      endif()
    else()
      set(MMU_X86_ISA_FLAGS -msse2)
    endif()
    set(MMU_X86_ISA_NAME SSE2)

    set(MMU_X86_ISA_FLAGS ${MMU_X86_ISA_FLAGS} PARENT_SCOPE)
    set(MMU_X86_ISA_NAME ${MMU_X86_ISA_NAME} PARENT_SCOPE)
    return()
  endif()

  # No usable ISA tier detected.
  set(MMU_X86_ISA_FLAGS PARENT_SCOPE)
  set(MMU_X86_ISA_NAME PARENT_SCOPE)
endfunction()
