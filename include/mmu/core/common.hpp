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

/* Check if a is greater or equal to b taking into account floating point noise
 *
 * Note that this function is assymmetric for the equality check as it uses
 * the scale of `b` to determine the tollerance.
 */
template <typename T1, typename T2, isFloat<T1> = true, isFloat<T2> = true>
inline bool greater_equal_tol(
    const T1 a,
    const T2 b,
    const double rtol = 1e-05,
    const double atol = 1e-8) {
    const double delta = a - b;
    const double scaled_tol = atol + rtol * b;
    // the first condition checks if a is greater than b given the tollerance
    // the second condition checks if a and b are approximately equal
    return delta > scaled_tol || std::abs(delta) <= scaled_tol;
}

template <typename T1, typename T2, isFloat<T1> = true, isFloat<T2> = true>
inline bool greq_tol(const T1 a, const T2 b) {
    return greater_equal_tol(a, b);
}

/* Check if a is equal to b taking into account floating point noise
 *
 * Note that this function is assymmetric for the equality check as it uses
 * the scale of `b` to determine the tollerance.
 */
template <typename T1, typename T2, isFloat<T1> = true, isFloat<T2> = true>
inline bool equal_tol(
    const T1 a,
    const T2 b,
    const double rtol = 1e-05,
    const double atol = 1e-8) {
    return std::abs(a - b) <= (atol + rtol * b);
}

inline int _linear_search(
    const double key,
    const double* arr,
    const int len,
    const int i0) {
    int i;
    for (i = i0; i < len && key >= arr[i]; i++)
        ;
    return i - 1;
}

template <typename T, isFloat<T> = true>
class LinearInterp {
    const int size;
    int j = 0;
    T left;
    T right;
    const T* dx;
    const T* dy;
    std::unique_ptr<T[]> slopes;

    // binary_search_with_guess(x_val, dx, lenxp, j);

    // adapted from numpy
    // https://github.com/numpy/numpy/blob/main/numpy/core/src/multiarray/compiled_base.c
    int binary_search_with_guess(const double key, int guess) {
        constexpr int LIKELY_IN_CACHE_SIZE = 8;
        int imin = 0;
        int imax = size;

        /* Handle keys outside of the arr range first */
        if (key > dx[size - 1]) {
            return size;
        } else if (key < dx[0]) {
            return -1;
        }

        /*
         * If len <= 4 use linear search.
         * From above we know key >= arr[0] when we start.
         */
        if (size <= 4) {
            return _linear_search(key, dx, size, 1);
        }

        if (guess > size - 3) {
            guess = size - 3;
        }
        if (guess < 1) {
            guess = 1;
        }

        /* check most likely values: guess - 1, guess, guess + 1 */
        if (key < dx[guess]) {
            if (key < dx[guess - 1]) {
                imax = guess - 1;
                /* last attempt to restrict search to items in cache */
                if (guess > LIKELY_IN_CACHE_SIZE
                    && key >= dx[guess - LIKELY_IN_CACHE_SIZE]) {
                    imin = guess - LIKELY_IN_CACHE_SIZE;
                }
            } else {
                /* key >= arr[guess - 1] */
                return guess - 1;
            }
        } else {
            /* key >= arr[guess] */
            if (key < dx[guess + 1]) {
                return guess;
            } else {
                /* key >= arr[guess + 1] */
                if (key < dx[guess + 2]) {
                    return guess + 1;
                } else {
                    /* key >= arr[guess + 2] */
                    imin = guess + 2;
                    /* last attempt to restrict search to items in cache */
                    if (guess < size - LIKELY_IN_CACHE_SIZE - 1
                        && key < dx[guess + LIKELY_IN_CACHE_SIZE]) {
                        imax = guess + LIKELY_IN_CACHE_SIZE;
                    }
                }
            }
        }

        /* finally, find index by bisection */
        while (imin < imax) {
            const int imid = imin + ((imax - imin) >> 1);
            if (key >= dx[imid]) {
                imin = imid + 1;
            } else {
                imax = imid;
            }
        }
        return imin - 1;
    }

   public:
    LinearInterp(int size, T left, T right, T* x, T* y)
        : size{size},
          left{left},
          right{right},
          dx{x},
          dy{y},
          slopes{std::unique_ptr<T[]>(new T[size - 1])} {
        for (int i = 0; i < size - 1; ++i) {
            slopes[i] = (dy[i + 1] - dy[i]) / (dx[i + 1] - dx[i]);
        }
    };

    T operator()(T x_val) {
        T res;
        j = binary_search_with_guess(x_val, j);
        if (j == -1) {
            res = left;
        } else if (j == size) {
            res = right;
        } else if (j == size - 1) {
            res = dy[j];
        } else {
            res = slopes[j] * (x_val - dx[j]) + dy[j];
        }
        return res;
    }
};

}  // namespace core
}  // namespace mmu

#endif  // INCLUDE_MMU_CORE_COMMON_HPP_
