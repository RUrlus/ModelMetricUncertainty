/* multn_loglike.hpp -- Implementation Multinomial uncertainty using Wilk's theorem
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_CORE_MULTN_LOGLIKE_HPP_
#define INCLUDE_MMU_CORE_MULTN_LOGLIKE_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cinttypes>
#include <limits>
#include <utility>

#include <mmu/core/common.hpp>
#include <mmu/core/mvn_error.hpp>

/* conf_mat layout:
 *  0 TN
 *  1 FP
 *  2 FN
 *  3 TP
 */

namespace mmu {
namespace core {

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

template<typename T>
inline double* linspace(const T start, T const end, const size_t steps, const bool inclusive = true) {
    if (steps == 0) {
        throw std::runtime_error("`steps` must be greater than zero.");
    } else if (steps == 1) {
        auto v = reinterpret_cast<double*>(std::malloc(steps * sizeof(double)));
        v[0] = static_cast<double>(start);
        return v;
    }
    auto values = reinterpret_cast<double*>(std::malloc(steps * sizeof(double)));
    double delta;
    double ub;
    if (inclusive) {
        delta = static_cast<double>(end - start) / static_cast<double>(steps - 1);
        values[steps - 1] = static_cast<double>(end);
    } else {
        delta = static_cast<double>(end - start) / static_cast<double>(steps);
        values[steps - 1] = static_cast<double>(end) - delta;
    }
    auto prev = static_cast<double>(start);
    values[0] = prev;
    values[steps - 1] = ub;
    const size_t N = steps - 1;
    for (size_t i = 1; i < N; ++i) {
        prev += delta;
        values[i] = prev;
    }
    return values;
}

/* Compute the most conservative probabilities for a given confusion matrix constrained by precision and recall.
 *
 * Parameters
 * ----------
 *  prec : precision
 *  rec : recall
 *  conf_mat : confusion matrix with order TN, FP, FN, TP
 *  probas : result array
 */
inline void constrained_fit_cmp(const double prec, const double rec, const int64_t* __restrict conf_mat, double* __restrict probas) {
    // n3 = TP + FP + FN
    const int64_t in3 = conf_mat[1] + conf_mat[2] + conf_mat[3];
    const auto n4 = static_cast<double>(conf_mat[0] + in3);
    const auto n3 = static_cast<double>(in3);

    // guard against divide by zero
    constexpr double ratio_fill = (1.0 - 1e-12) / 1e-12;
    const double rec_ratio = rec > std::numeric_limits<double>::epsilon() ? (1.0 - rec) / rec : ratio_fill;
    const double prec_ratio = prec > std::numeric_limits<double>::epsilon() ? (1.0 - prec) / prec : ratio_fill;
    const double alpha = 1 + prec_ratio + rec_ratio;
    const double p_tp = (n3 / n4) * (1. / alpha);
    const double p_fn = rec_ratio * p_tp;
    const double p_fp = prec_ratio * p_tp;
    // guard against floating point noise resulting in negative probabilities
    const double p_tn = std::max(1. - p_fn - p_fp - p_tp, 0.0);
    probas[0] = p_tn;
    probas[1] = p_fp;
    probas[2] = p_fn;
    probas[3] = p_tp;
} // constrained_fit_cmp

/* Compute the most conservative probabilities for a given confusion matrix constrained by precision and recall.
 *
 * Parameters
 * ----------
 *  prec : precision
 *  rec : recall
 *  conf_mat : confusion matrix with order TN, FP, FN, TP
 *  probas : result array
 */
inline void constrained_fit_cmp(const double prec, const double rec, const double n3, const double n4, double* __restrict probas) {
    // guard against divide by zero
    constexpr double ratio_fill = (1.0 - 1e-12) / 1e-12;
    const double rec_ratio = rec > std::numeric_limits<double>::epsilon() ? (1.0 - rec) / rec : ratio_fill;
    const double prec_ratio = prec > std::numeric_limits<double>::epsilon() ? (1.0 - prec) / prec : ratio_fill;
    const double alpha = 1 + prec_ratio + rec_ratio;
    const double p_tp = (n3 / n4) * (1. / alpha);
    const double p_fn = rec_ratio * p_tp;
    const double p_fp = prec_ratio * p_tp;
    // guard against floating point noise resulting in negative probabilities
    const double p_tn = std::max(1. - p_fn - p_fp - p_tp, 0.0);
    probas[0] = p_tn;
    probas[1] = p_fp;
    probas[2] = p_fn;
    probas[3] = p_tp;
} // constrained_fit_cmp

/* Compute -2logp of multinomial distribution given a precision and recall.
 *
 * Step 1: fit with all parameters free
 * Step 2: fit multinomial with fixed recall and precision
 * Step 3: compute -2ln(L_h0 / L_h1)
 *
 * We compute -2ln(L_h0) + 2ln(L_h1))
 *
 */
inline double wilks_loglike_diff(const double prec, const double rec, const int64_t* __restrict conf_mat) {
    const int64_t in3 = conf_mat[1] + conf_mat[2] + conf_mat[3];
    const auto n3 = static_cast<double>(in3);
    const auto n4 = static_cast<double>(conf_mat[0] + in3);

    // constitutes the optimal/unconstrained fit of a multinomial
    const double x_tn = static_cast<double>(conf_mat[0]);
    const double h0_p_tn = x_tn / n4;
    const double x_fp = static_cast<double>(conf_mat[1]);
    const double h0_p_fp = x_fp / n4;
    const double x_fn = static_cast<double>(conf_mat[2]);
    const double h0_p_fn = x_fn / n4;
    const double x_tp = static_cast<double>(conf_mat[3]);
    const double h0_p_tp = x_tp / n4;

    const double nll_h0 = (
        -2. * xlogy(x_tn, h0_p_tn)
        - 2. * xlogy(x_fp, h0_p_fp)
        - 2. * xlogy(x_fn, h0_p_fn)
        - 2. * xlogy(x_tp, h0_p_tp)
    );

    std::array<double, 4> p_h1;
    constrained_fit_cmp(prec, rec, n3, n4, p_h1.begin());

    const double nll_h1 = (
        -2. * xlogy(x_tn, p_h1[0])
        - 2. * xlogy(x_fp, p_h1[1])
        - 2. * xlogy(x_fn, p_h1[2])
        - 2. * xlogy(x_tp, p_h1[3])
    );
    return nll_h0 - nll_h1;
}  // wilks_loglike_diff

