/* erfinv.hpp -- Implementation of inverse error function.
 * Copyright 2021 Ralph Urlus
 *
 * Based on:
 * Wichura, M. J. (1988). Algorithm AS 241: The Percentage Points of the Normal Distribution. Journal of the Royal
 * Statistical Society. Series C (Applied Statistics), 37(3), 477-484. https://doi.org/10.2307/2347330
 *
 * Adapted from: https://github.com/golang/go/blob/master/src/math/erfinv.go#L7
 */
#ifndef INCLUDE_MMU_CORE_ERFINV_HPP_
#define INCLUDE_MMU_CORE_ERFINV_HPP_

#include <cmath>   // for sqrt
#include <limits>  // for numeric_limits

#include <mmu/core/common.hpp>

namespace mmu {
namespace core {

/* Inverse of the floating-point error function.
 *
 * Based on:
 * Wichura, M. J. (1988). Algorithm AS 241: The Percentage Points of the Normal Distribution. Journal of the Royal
 * Statistical Society. Series C (Applied Statistics), 37(3), 477-484. https://doi.org/10.2307/2347330
 *
 * Adapted from: https://github.com/golang/go/blob/master/src/math/erfinv.go#L7
 * Special cases are:
 * x == 1   -> infinity
 * x == -1  -> neg. infinity
 * x < -1   -> NaN
 * x > 1    -> NaN
 * x == NaN -> NaN
 */
template <typename T, isFloat<T> = true>
inline double erfinv(T x) {
    // special cases
    constexpr double epsilon = std::numeric_limits<double>::epsilon();
    // x == 1
    if (abs(x - 1.) < epsilon) {
        return std::numeric_limits<double>::infinity();
    }
    // x == -1
    if (abs(x + 1.) < epsilon) {
        return -1 * std::numeric_limits<double>::infinity();
    }
    if (std::isnan(x) || x < -1.0 || x > 1.0) {
        return std::numeric_limits<double>::signaling_NaN();
    }

    const double a0 = 1.1975323115670912564578e0;
    const double a1 = 4.7072688112383978012285e1;
    const double a2 = 6.9706266534389598238465e2;
    const double a3 = 4.8548868893843886794648e3;
    const double a4 = 1.6235862515167575384252e4;
    const double a5 = 2.3782041382114385731252e4;
    const double a6 = 1.1819493347062294404278e4;
    const double a7 = 8.8709406962545514830200e2;
    const double b0 = 1.0000000000000000000e0;
    const double b1 = 4.2313330701600911252e1;
    const double b2 = 6.8718700749205790830e2;
    const double b3 = 5.3941960214247511077e3;
    const double b4 = 2.1213794301586595867e4;
    const double b5 = 3.9307895800092710610e4;
    const double b6 = 2.8729085735721942674e4;
    const double b7 = 5.2264952788528545610e3;
    // Coefficients for approximation to erf in 0.85 < |x| <= 1-2*exp(-25)
    const double c0 = 1.42343711074968357734e0;
    const double c1 = 4.63033784615654529590e0;
    const double c2 = 5.76949722146069140550e0;
    const double c3 = 3.64784832476320460504e0;
    const double c4 = 1.27045825245236838258e0;
    const double c5 = 2.41780725177450611770e-1;
    const double c6 = 2.27238449892691845833e-2;
    const double c7 = 7.74545014278341407640e-4;
    const double d0 = 1.4142135623730950488016887e0;
    const double d1 = 2.9036514445419946173133295e0;
    const double d2 = 2.3707661626024532365971225e0;
    const double d3 = 9.7547832001787427186894837e-1;
    const double d4 = 2.0945065210512749128288442e-1;
    const double d5 = 2.1494160384252876777097297e-2;
    const double d6 = 7.7441459065157709165577218e-4;
    const double d7 = 1.4859850019840355905497876e-9;
    // Coefficients for approximation to erf in 1-2*exp(-25) < |x| < 1
    const double e0 = 6.65790464350110377720e0;
    const double e1 = 5.46378491116411436990e0;
    const double e2 = 1.78482653991729133580e0;
    const double e3 = 2.96560571828504891230e-1;
    const double e4 = 2.65321895265761230930e-2;
    const double e5 = 1.24266094738807843860e-3;
    const double e6 = 2.71155556874348757815e-5;
    const double e7 = 2.01033439929228813265e-7;
    const double f0 = 1.414213562373095048801689e0;
    const double f1 = 8.482908416595164588112026e-1;
    const double f2 = 1.936480946950659106176712e-1;
    const double f3 = 2.103693768272068968719679e-2;
    const double f4 = 1.112800997078859844711555e-3;
    const double f5 = 2.611088405080593625138020e-5;
    const double f6 = 2.010321207683943062279931e-7;
    const double f7 = 2.891024605872965461538222e-15;

    const double ln2 = 0.693147180559945309417;

    bool sign = false;
    if (x < 0) {
        x = -1 * x;
        sign = true;
    }

    double result, r, z1, z2;
    if (x <= 0.85) {  // |x| <= 0.85
        r = 0.180625 - 0.25 * x * x;
        z1 = ((((((a7 * r + a6) * r + a5) * r + a4) * r + a3) * r + a2) * r + a1) * r + a0;
        z2 = ((((((b7 * r + b6) * r + b5) * r + b4) * r + b3) * r + b2) * r + b1) * r + b0;
        result = (x * z1) / z2;
    } else {
        r = std::sqrt(ln2 - std::log(1.0 - x));
        if (r <= 5.0) {
            r -= 1.6;
            z1 = ((((((c7 * r + c6) * r + c5) * r + c4) * r + c3) * r + c2) * r + c1) * r + c0;
            z2 = ((((((d7 * r + d6) * r + d5) * r + d4) * r + d3) * r + d2) * r + d1) * r + d0;
        } else {
            r -= 5.0;
            z1 = ((((((e7 * r + e6) * r + e5) * r + e4) * r + e3) * r + e2) * r + e1) * r + e0;
            z2 = ((((((f7 * r + f6) * r + f5) * r + f4) * r + f3) * r + f2) * r + f1) * r + f0;
        }
        result = z1 / z2;
    }

    if (sign) {
        return -1 * result;
    }
    return result;
}

}  // namespace core
}  // namespace mmu

#endif  // INCLUDE_MMU_CORE_ERFINV_HPP_
