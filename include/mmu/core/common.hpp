/* common.hpp -- Utility functions and macros used in multiple headers.
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_CORE_COMMON_HPP_
#define INCLUDE_MMU_CORE_COMMON_HPP_

#define UNUSED(x) (void)(x)

#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
#define OS_WIN
#endif

// handle error C2059: syntax error: ';'  on windows for this Macro
#if not defined(OS_WIN)
  #define STRINGIFY(x) #x
  #define MACRO_STRINGIFY(x) STRINGIFY(x)
#endif

// Fix for lack of ssize_t on Windows for CPython3.10
#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif


#include <type_traits>

namespace mmu {

template<typename T>
using isInt = std::enable_if_t<std::is_integral<T>::value, bool>;

template<typename T>
using isFloat = std::enable_if_t<std::is_floating_point<T>::value, bool>;

}  // namespace mmu

#endif  // INCLUDE_MMU_CORE_COMMON_HPP_
