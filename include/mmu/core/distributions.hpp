/* mmu/random/distributions.hpp -- Headers of binomial and multinomial
 * generators.
 *
 * The below implementation was taken and modified from Numpy:
 *
 * Copyright (c) 2005-2022, NumPy Developers.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials provided
 *        with the distribution.
 *
 *     * Neither the name of the NumPy Developers nor the names of any
 *        contributors may be used to endorse or promote products derived
 *        from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef INCLUDE_MMU_CORE_DISTRIBUTIONS_HPP_
#define INCLUDE_MMU_CORE_DISTRIBUTIONS_HPP_

#include <cinttypes>
#include <mmu/core/common.hpp>

namespace mmu {
namespace core {
namespace random {
namespace details {

#ifndef MIN
#define MIN(x, y) (((x) < (y)) ? x : y)
#define MAX(x, y) (((x) > (y)) ? x : y)
#endif

typedef struct s_binomial_t {
    int has_binomial; /* !=0: following parameters initialized for binomial */
    double psave;
    int64_t nsave;
    double r;
    double q;
    double fm;
    int64_t m;
    double p1;
    double xm;
    double xl;
    double xr;
    double c;
    double laml;
    double lamr;
    double p2;
    double p3;
    double p4;
} binomial_t;

inline double uint64_to_double(uint64_t rnd) {
    constexpr double rhs = 1.0 / 9007199254740992.0;
    return (rnd >> 11) * rhs;
}

inline double next_double(pcg64_dxsm& rng) {
    constexpr double rhs = 1.0 / 9007199254740992.0;
    return (rng() >> 11) * rhs;
}

inline double random_standard_uniform(pcg64_dxsm& rng) {
    return next_double(rng);
}

inline void
random_standard_uniform_fill(pcg64_dxsm& rng, int64_t cnt, double* out) {
    int64_t i;
    for (i = 0; i < cnt; i++) {
        out[i] = next_double(rng);
    }
}

inline int64_t random_binomial_btpe(
    pcg64_dxsm& rng,
    const int64_t n,
    const double p,
    binomial_t* binomial) {
    double r, q, fm, p1, xm, xl, xr, c, laml, lamr, p2, p3, p4;
    double a, u, v, s, F, rho, t, A, nrq, x1, x2, f1, f2, z, z2, w, w2, x;
    int64_t m, y, k, i;

    if (!(binomial->has_binomial) || (binomial->nsave != n)
        || (binomial->psave != p)) {
        /* initialize */
        binomial->nsave = n;
        binomial->psave = p;
        binomial->has_binomial = 1;
        binomial->r = r = MIN(p, 1.0 - p);
        binomial->q = q = 1.0 - r;
        binomial->fm = fm = n * r + r;
        binomial->m = m = (int64_t)floor(binomial->fm);
        binomial->p1 = p1 = floor(2.195 * sqrt(n * r * q) - 4.6 * q) + 0.5;
        binomial->xm = xm = m + 0.5;
        binomial->xl = xl = xm - p1;
        binomial->xr = xr = xm + p1;
        binomial->c = c = 0.134 + 20.5 / (15.3 + m);
        a = (fm - xl) / (fm - xl * r);
        binomial->laml = laml = a * (1.0 + a / 2.0);
        a = (xr - fm) / (xr * q);
        binomial->lamr = lamr = a * (1.0 + a / 2.0);
        binomial->p2 = p2 = p1 * (1.0 + 2.0 * c);
        binomial->p3 = p3 = p2 + c / laml;
        binomial->p4 = p4 = p3 + c / lamr;
    } else {
        r = binomial->r;
        q = binomial->q;
        fm = binomial->fm;
        m = binomial->m;
        p1 = binomial->p1;
        xm = binomial->xm;
        xl = binomial->xl;
        xr = binomial->xr;
        c = binomial->c;
        laml = binomial->laml;
        lamr = binomial->lamr;
        p2 = binomial->p2;
        p3 = binomial->p3;
        p4 = binomial->p4;
    }

/* sigh ... */
Step10:
    nrq = n * r * q;
    u = next_double(rng) * p4;
    v = next_double(rng);
    if (u > p1)
        goto Step20;
    y = (int64_t)floor(xm - p1 * v + u);
    goto Step60;

Step20:
    if (u > p2)
        goto Step30;
    x = xl + (u - p1) / c;
    v = v * c + 1.0 - fabs(m - x + 0.5) / p1;
    if (v > 1.0)
        goto Step10;
    y = (int64_t)floor(x);
    goto Step50;

Step30:
    if (u > p3)
        goto Step40;
    y = (int64_t)floor(xl + log(v) / laml);
    /* Reject if v==0.0 since previous cast is undefined */
    if ((y < 0) || (v == 0.0))
        goto Step10;
    v = v * (u - p2) * laml;
    goto Step50;

Step40:
    y = (int64_t)floor(xr - log(v) / lamr);
    /* Reject if v==0.0 since previous cast is undefined */
    if ((y > n) || (v == 0.0))
        goto Step10;
    v = v * (u - p3) * lamr;

Step50:
    k = llabs(y - m);
    if ((k > 20) && (k < ((nrq) / 2.0 - 1)))
        goto Step52;

    s = r / q;
    a = s * (n + 1);
    F = 1.0;
    if (m < y) {
        for (i = m + 1; i <= y; i++) {
            F *= (a / i - s);
        }
    } else if (m > y) {
        for (i = y + 1; i <= m; i++) {
            F /= (a / i - s);
        }
    }
    if (v > F)
        goto Step10;
    goto Step60;

Step52:
    rho = (k / (nrq))
          * ((k * (k / 3.0 + 0.625) + 0.16666666666666666) / nrq + 0.5);
    t = -k * k / (2 * nrq);
    /* log(0.0) ok here */
    A = log(v);
    if (A < (t - rho))
        goto Step60;
    if (A > (t + rho))
        goto Step10;

    x1 = y + 1;
    f1 = m + 1;
    z = n + 1 - m;
    w = n - y + 1;
    x2 = x1 * x1;
    f2 = f1 * f1;
    z2 = z * z;
    w2 = w * w;
    if (A
        > (xm * log(f1 / x1) + (n - m + 0.5) * log(z / w)
           + (y - m) * log(w * r / (x1 * q))
           + (13680. - (462. - (132. - (99. - 140. / f2) / f2) / f2) / f2) / f1
                 / 166320.
           + (13680. - (462. - (132. - (99. - 140. / z2) / z2) / z2) / z2) / z
                 / 166320.
           + (13680. - (462. - (132. - (99. - 140. / x2) / x2) / x2) / x2) / x1
                 / 166320.
           + (13680. - (462. - (132. - (99. - 140. / w2) / w2) / w2) / w2) / w
                 / 166320.)) {
        goto Step10;
    }

Step60:
    if (p > 0.5) {
        y = n - y;
    }

    return y;
}

inline int64_t random_binomial_inversion(
    pcg64_dxsm& rng,
    const int64_t n,
    const double p,
    binomial_t* binomial) {
    double q, qn, np, px, U;
    int64_t X, bound;

    if (!(binomial->has_binomial) || (binomial->nsave != n)
        || (binomial->psave != p)) {
        binomial->nsave = n;
        binomial->psave = p;
        binomial->has_binomial = 1;
        binomial->q = q = 1.0 - p;
        binomial->r = qn = exp(n * log(q));
        binomial->c = np = n * p;
        binomial->m = bound = (int64_t)MIN(n, np + 10.0 * sqrt(np * q + 1));
    } else {
        q = binomial->q;
        qn = binomial->r;
        np = binomial->c;
        bound = binomial->m;
    }
    X = 0;
    px = qn;
    U = next_double(rng);
    while (U > px) {
        X++;
        if (X > bound) {
            X = 0;
            px = qn;
            U = next_double(rng);
        } else {
            U -= px;
            px = ((n - X + 1) * p * px) / (X * q);
        }
    }
    return X;
}

inline int64_t random_binomial(
    pcg64_dxsm& rng,
    const double p,
    const int64_t n,
    binomial_t* binomial) {
    double q;

    if ((n == 0LL) || (p == 0.0f))
        return 0;

    if (p <= 0.5) {
        if (p * n <= 30.0) {
            return random_binomial_inversion(rng, n, p, binomial);
        } else {
            return random_binomial_btpe(rng, n, p, binomial);
        }
    } else {
        q = 1.0 - p;
        if (q * n <= 30.0) {
            return n - random_binomial_inversion(rng, n, q, binomial);
        } else {
            return n - random_binomial_btpe(rng, n, q, binomial);
        }
    }
}

inline void random_multinomial(
    pcg64_dxsm& rng,
    const int64_t n,
    int64_t* mnix,
    const double* pix,
    const int64_t d,
    binomial_t* binomial) {
    double remaining_p = 1.0;
    int64_t j;
    int64_t dn = n;
    for (j = 0; j < (d - 1); j++) {
        mnix[j] = random_binomial(rng, pix[j] / remaining_p, dn, binomial);
        dn = dn - mnix[j];
        if (dn <= 0) {
            break;
        }
        remaining_p -= pix[j];
    }
    if (dn > 0) {
        mnix[d - 1] = dn;
    }
}

}  // namespace details
}  // namespace random
}  // namespace core
}  // namespace mmu

#endif  // INCLUDE_MMU_CORE_DISTRIBUTIONS_HPP_
