/* common.hpp -- Utility functions and macros used in multiple headers.
 * Copyright 2022 Ralph Urlus
 */
#pragma once

#include <cmath>
#include <cstddef>
#include <cstring>  // for memset
#include <limits>
#include <type_traits>

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

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4127)
#include <BaseTsd.h>
#pragma warning(pop)
#endif

namespace mmu {

template <typename T>
using isInt = std::enable_if_t<std::is_integral<T>::value, bool>;

template <typename T>
using isFloat = std::enable_if_t<std::is_floating_point<T>::value, bool>;

namespace core {

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
