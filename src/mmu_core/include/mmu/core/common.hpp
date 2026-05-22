/* common.hpp -- Utility functions and macros used in multiple headers.
 * Copyright 2022 Ralph Urlus
 */
#ifndef INCLUDE_MMU_CORE_COMMON_HPP_
#define INCLUDE_MMU_CORE_COMMON_HPP_

#define UNUSED(x) (void)(x)

#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) \
    || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
#define OS_WIN
#endif

// handle error C2059: syntax error: ';'  on windows for this Macro
#ifndef OS_WIN
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
#endif

// Fix for lack of ssize_t on Windows for CPython3.10
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning( \
    disable : 4127)  // warning C4127: Conditional expression is constant
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#include <cstring>  // for memset
#include <random>
#include <type_traits>

#include <pcg_extras.hpp>
#include <pcg_random.hpp>

namespace mmu {

template <typename T>
using isInt = std::enable_if_t<std::is_integral<T>::value, bool>;

template <typename T>
using isFloat = std::enable_if_t<std::is_floating_point<T>::value, bool>;

namespace core {

// fill value for the chi2 scres, this values results
// p-values very close to 1
// chi2.ppf(1-1e-14) --> 64.47398179869367
constexpr double MULT_DEFAULT_CHI2_SCORE = 65.0;

namespace random {

typedef pcg_engines::setseq_dxsm_128_64 pcg64_dxsm;
typedef pcg_extras::seed_seq_from<std::random_device> pcg_seed_seq;

}  // namespace random

namespace details {

/* clamp value between lo and hi */
template <typename T>
inline const T& clamp(const T& v, const T& lo, const T& hi) {
    return v < lo ? lo : v > hi ? hi : v;
}

template <typename T, isFloat<T> = true>
inline double xlogy(T x, T y) {
    if ((x <= std::numeric_limits<T>::epsilon()) && (!std::isnan(y))) {
        return 0.0;
    }
    return x * std::log(y);
}

template <typename T, isInt<T> = true>
inline double xlogy(T x, T y) {
    if (x == 0) {
        return 0.0;
    }
    return static_cast<double>(x) * std::log(static_cast<double>(y));
}

}  // namespace details

template <typename T>
inline void zero_array(T* ptr, size_t n_elem) {
    // zero the memory
    memset(ptr, 0, n_elem * sizeof(T));
}

template <typename T, const size_t n_elem>
inline void zero_array(T* ptr) {
    // zero the memory
    memset(ptr, 0, n_elem * sizeof(T));
}

}  // namespace core

}  // namespace mmu

#endif  // INCLUDE_MMU_CORE_COMMON_HPP_
