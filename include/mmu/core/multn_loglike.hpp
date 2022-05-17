/* multn_loglike.hpp -- Implementation Multinomial uncertainty using Wilk's theorem
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_CORE_MULTN_LOGLIKE_HPP_
#define INCLUDE_MMU_CORE_MULTN_LOGLIKE_HPP_

#if defined(MMU_HAS_OPENMP_SUPPORT)
#include <omp.h>
#endif  // MMU_HAS_OPENMP_SUPPORT

#include <algorithm>
#include <array>
#include <cmath>
#include <cinttypes>
#include <limits>
#include <memory>
#include <utility>

#include <mmu/core/common.hpp>
#include <mmu/core/mvn_error.hpp>
#include <mmu/core/random.hpp>
#include <mmu/core/metrics.hpp>

/* conf_mat layout:
 *  0 TN
 *  1 FP
 *  2 FN
 *  3 TP
 */

namespace mmu {
namespace core {
namespace details {

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
        auto v = new double[steps];
        v[0] = static_cast<double>(start);
        return v;
    }
    auto values = new double[steps];
    double delta;
    if (inclusive) {
        delta = static_cast<double>(end - start) / static_cast<double>(steps - 1);
        values[steps - 1] = static_cast<double>(end);
    } else {
        delta = static_cast<double>(end - start) / static_cast<double>(steps);
        values[steps - 1] = static_cast<double>(end) - delta;
    }
    auto prev = static_cast<double>(start);
    values[0] = prev;
    const size_t N = steps - 1;
    for (size_t i = 1; i < N; ++i) {
        prev += delta;
        values[i] = prev;
    }
    return values;
}

inline double* linspace(const double start, double const end, const size_t steps, const double delta) {
    auto values = new double[steps];
    double prev = start;
    values[0] = start;
    values[steps - 1] = end;
    const size_t N = steps - 1;
    for (size_t i = 1; i < N - 1; ++i) {
        prev += delta;
        values[i] = prev;
    }
    return values;
}

template<typename T>
inline double linspace_for_gen(const T start, T const end, const size_t steps) {
    if (steps < 2) {
        throw std::runtime_error("`steps` must be greater than 1.");
    }
    return static_cast<double>(end - start) / static_cast<double>(steps - 1);
}

/* Determine grid for precision and recall based on their marginal std
 * deviations assuming a Multivariate Normal
 *
 *
 * Returns
 * -------
 * a = precision range
 * b = recall range
 *
 * Note that both need to freed
 */
inline std::pair<double*, double*> get_pr_grid(
    const size_t n_bins,
    const int64_t* __restrict conf_mat,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4
) {
    const double max_prec_clip = conf_mat[1] == 0 ? 0.0 : epsilon;
    const double max_rec_clip = conf_mat[2] == 0 ? 0.0 : epsilon;
    // computes prec, prec_sigma, rec, rec_sigma accounting for edge cases
    std::array<double, 4> prec_rec;
    pr_mvn_sigma(conf_mat, prec_rec.data());

    const double ns_prec_sigma = n_sigmas * prec_rec[1];
    const double ns_rec_sigma = n_sigmas * prec_rec[3];

    const double prec_max = std::min(prec_rec[0] + ns_prec_sigma, 1 - max_prec_clip);
    const double prec_min = std::max(prec_rec[0] - ns_prec_sigma, epsilon);
    double* prec_range = details::linspace(prec_min, prec_max, n_bins);

    const double rec_max = std::min(prec_rec[2] + ns_rec_sigma, 1. - max_rec_clip);
    const double rec_min = std::max(prec_rec[2] - ns_rec_sigma, epsilon);
    double* rec_range = details::linspace(rec_min, rec_max, n_bins);

    return std::make_pair(prec_range, rec_range);
}  // get_pr_grid

inline void get_pr_grid_delta(
    const size_t n_bins,
    const int64_t* __restrict conf_mat,
    double* result,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4
) {
    const double max_prec_clip = conf_mat[1] == 0 ? 0.0 : epsilon;
    const double max_rec_clip = conf_mat[2] == 0 ? 0.0 : epsilon;
    // computes prec, prec_sigma, rec, rec_sigma accounting for edge cases
    std::array<double, 4> prec_rec;
    pr_mvn_sigma(conf_mat, prec_rec.data());

    const double ns_prec_sigma = n_sigmas * prec_rec[1];
    const double ns_rec_sigma = n_sigmas * prec_rec[3];

    const double prec_max = std::min(prec_rec[0] + ns_prec_sigma, 1 - max_prec_clip);
    const double prec_min = std::max(prec_rec[0] - ns_prec_sigma, epsilon);
    result[0] = prec_min;
    result[1] = details::linspace_for_gen(prec_min, prec_max, n_bins);
    result[2] = prec_max;

    const double rec_max = std::min(prec_rec[2] + ns_rec_sigma, 1. - max_rec_clip);
    const double rec_min = std::max(prec_rec[2] - ns_rec_sigma, epsilon);
    result[3] = rec_min;
    result[4] = details::linspace_for_gen(rec_min, rec_max, n_bins);
    result[5] = rec_max;
}  // get_pr_grid

class PrGridBounds {
    const int64_t n_prec_bins;
    const int64_t n_rec_bins;
    const double n_sigmas;
    const double epsilon;
    double max_prec_clip;
    double max_rec_clip;
    double prec_max;
    double prec_min;
    double rec_max;
    double rec_min;
    double ns_prec_sigma;
    double ns_rec_sigma;
    const double* __restrict precs;
    const double* __restrict recs;
    std::array<double, 4> prec_rec;

 public:
    int64_t prec_idx_min = 0;
    int64_t prec_idx_max = 0;
    int64_t rec_idx_min = 0;
    int64_t rec_idx_max = 0;

    PrGridBounds(
        const int64_t n_prec_bins,
        const int64_t n_rec_bins,
        const double n_sigmas,
        const double epsilon,
        const double* __restrict precs,
        const double* __restrict recs
    ) : n_prec_bins{n_prec_bins},
        n_rec_bins{n_rec_bins},
        n_sigmas{n_sigmas},
        epsilon{epsilon},
        precs{precs},
        recs{recs}
    {}

    void compute_bounds(const int64_t* __restrict conf_mat) {
        max_prec_clip = conf_mat[1] == 0 ? 0.0 : epsilon;
        max_rec_clip = conf_mat[2] == 0 ? 0.0 : epsilon;
        // computes prec, prec_sigma, rec, rec_sigma accounting for edge cases
        pr_mvn_sigma(conf_mat, prec_rec.data());

        ns_prec_sigma = n_sigmas * prec_rec[1];
        prec_max = std::min(prec_rec[0] + ns_prec_sigma, 1 - max_prec_clip);
        prec_min = std::max(prec_rec[0] - ns_prec_sigma, epsilon);
        prec_idx_min = 0;
        prec_idx_max = n_prec_bins;

        int64_t i;
        for (i = 0; i < n_prec_bins; i++) {
            if (prec_min < precs[i]) {
                prec_idx_min = i - 1;
                break;
            }
        }
        prec_idx_min = prec_idx_min > 0 ? prec_idx_min : 0;

        for (i = prec_idx_min; i < n_rec_bins; i++) {
            if (prec_max < precs[i]) {
                prec_idx_max = i + 1;
                break;
            }
        }
        prec_idx_max = prec_idx_max <= n_prec_bins ? prec_idx_max : n_prec_bins;

        ns_rec_sigma = n_sigmas * prec_rec[3];
        rec_max = std::min(prec_rec[2] + ns_rec_sigma, 1. - max_rec_clip);
        rec_min = std::max(prec_rec[2] - ns_rec_sigma, epsilon);
        rec_idx_min = 0;
        rec_idx_max = n_rec_bins;
        for (i = 0; i < n_rec_bins; i++) {
            if (rec_min < recs[i]) {
                rec_idx_min = i - 1;
                break;
            }
        }
        rec_idx_min = rec_idx_min > 0 ? rec_idx_min : 0;

        for (i = rec_idx_min; i < n_rec_bins; i++) {
            if (rec_max < recs[i]) {
                rec_idx_max = i + 1;
                break;
            }
        }
        rec_idx_max = rec_idx_max <= n_rec_bins ? rec_idx_max : n_rec_bins;
    }
};

inline void get_pr_grid_bounds(
    const int64_t prec_bins,
    const int64_t rec_bins,
    const int64_t* __restrict conf_mat,
    const double* __restrict precs,
    const double* __restrict recs,
    int64_t* result,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4
) {
    const double max_prec_clip = conf_mat[1] == 0 ? 0.0 : epsilon;
    const double max_rec_clip = conf_mat[2] == 0 ? 0.0 : epsilon;
    // computes prec, prec_sigma, rec, rec_sigma accounting for edge cases
    std::array<double, 4> prec_rec;
    pr_mvn_sigma(conf_mat, prec_rec.data());

    const double ns_prec_sigma = n_sigmas * prec_rec[1];
    const double prec_max = std::min(prec_rec[0] + ns_prec_sigma, 1 - max_prec_clip);
    const double prec_min = std::max(prec_rec[0] - ns_prec_sigma, epsilon);
    int64_t prec_idx_min = 0;
    int64_t prec_idx_max = prec_bins;
    for (int64_t i = 0; i < prec_bins; i++) {
        if (prec_min < precs[i]) {
            prec_idx_min = i - 1;
            break;
        }
    }
    result[0] = prec_idx_min > 0 ? prec_idx_min : 0;

    for (int64_t i = prec_idx_min; i < rec_bins; i++) {
        if (prec_max < precs[i]) {
            prec_idx_max = i + 1;
            break;
        }
    }
    result[1] = prec_idx_max <= prec_bins ? prec_idx_max : prec_bins;

    const double ns_rec_sigma = n_sigmas * prec_rec[3];
    const double rec_max = std::min(prec_rec[2] + ns_rec_sigma, 1. - max_rec_clip);
    const double rec_min = std::max(prec_rec[2] - ns_rec_sigma, epsilon);
    int64_t rec_idx_min = 0;
    int64_t rec_idx_max = rec_bins;
    for (int64_t i = 0; i < rec_bins; i++) {
        if (rec_min < recs[i]) {
            rec_idx_min = i - 1;
            break;
        }
    }
    result[2] = rec_idx_min > 0 ? rec_idx_min : 0;

    for (int64_t i = rec_idx_min; i < rec_bins; i++) {
        if (rec_max < recs[i]) {
            rec_idx_max = i + 1;
            break;
        }
    }
    result[3] = rec_idx_max <= rec_bins ? rec_idx_max : rec_bins;
}  // get_pr_grid_bounds

}  // namespace details

/* Compute the most conservative probabilities for a given confusion matrix constrained by precision and recall.
 *
 * equivalent of phat in python dev implementation
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
    const double rec_ratio = (1.0 - rec) / rec;
    const double prec_ratio = (1.0 - prec) / prec;
    probas[3] = (n3 / n4) * (1. / (1. + prec_ratio + rec_ratio));
    probas[2] = rec_ratio * probas[3];
    probas[1] = prec_ratio * probas[3];
    // guard against floating point noise resulting in negative probabilities
    probas[0] = std::max(1. - probas[1] - probas[2] - probas[3], 0.0);
} // constrained_fit_cmp

typedef struct s_prof_loglike_t {
    int64_t in;
    double n3;
    double n;
    double x_tn;
    double x_fp;
    double x_fn;
    double x_tp;
    double p_tn;
    double p_fp;
    double p_fn;
    double p_tp;
    double nll_h0;
} prof_loglike_t;

inline void set_prof_loglike_store(
    const int64_t* __restrict conf_mat,
    prof_loglike_t* store
) {
    // total number of entries in the confusion matrix
    const int64_t in3 = conf_mat[1] + conf_mat[2] + conf_mat[3];
    store->n3 = static_cast<double>(in3);
    store->in = conf_mat[0] + in3;
    store->n = static_cast<double>(store->in);

    store->x_tn = static_cast<double>(conf_mat[0]);
    store->x_fp = static_cast<double>(conf_mat[1]);
    store->x_fn = static_cast<double>(conf_mat[2]);
    store->x_tp = static_cast<double>(conf_mat[3]);

    store->p_tn = store->x_tn / store->n;
    store->p_fp = store->x_fp / store->n;
    store->p_fn = store->x_fn / store->n;
    store->p_tp = store->x_tp / store->n;

    store->nll_h0 = (
        -2. * details::xlogy(store->x_tn, store->p_tn)
        - 2. * details::xlogy(store->x_fp, store->p_fp)
        - 2. * details::xlogy(store->x_fn, store->p_fn)
        - 2. * details::xlogy(store->x_tp, store->p_tp)
    );
}

/* Compute -2logp of multinomial distribution given a precision and recall.
 *
 * Step 1: fit with all parameters free
 * Step 2: fit multinomial with fixed recall and precision
 * Step 3: compute -2ln(L_h0 / L_h1)
 *
 * We compute -2ln(L_h0) + 2ln(L_h1))
 */
inline double prof_loglike(
    const double prec,
    const double rec,
    prof_loglike_t* store,
    double* p_h1
) {
    // constitutes the optimal/unconstrained fit of a multinomial
    constrained_fit_cmp(prec, rec, store->n3, store->n, p_h1);

    const double nll_h1 = (
        -2. * details::xlogy(store->x_tn, p_h1[0])
        - 2. * details::xlogy(store->x_fp, p_h1[1])
        - 2. * details::xlogy(store->x_fn, p_h1[2])
        - 2. * details::xlogy(store->x_tp, p_h1[3])
    );
    return nll_h1 - store->nll_h0;
}  // prof_loglike

/* Compute -2logp of multinomial distribution given a precision and recall.
 *
 * Step 1: fit with all parameters free
 * Step 2: fit multinomial with fixed recall and precision
 * Step 3: compute -2ln(L_h0 / L_h1)
 *
 * We compute -2ln(L_h0) + 2ln(L_h1))
 */
inline double prof_loglike(
    const double prec,
    const double rec,
    const double n,
    const int64_t* __restrict conf_mat,
    double* p_h1
) {
    const auto n3 = n - static_cast<double>(conf_mat[0]);
    const auto x_tn = static_cast<double>(conf_mat[0]);
    const auto x_fp = static_cast<double>(conf_mat[1]);
    const auto x_fn = static_cast<double>(conf_mat[2]);
    const auto x_tp = static_cast<double>(conf_mat[3]);

    // constitutes the optimal/unconstrained fit of a multinomial
    const double nll_h0 = (
        -2. * details::xlogy(x_tn, x_tn / n)
        - 2. * details::xlogy(x_fp, x_fp / n)
        - 2. * details::xlogy(x_fn, x_fn / n)
        - 2. * details::xlogy(x_tp, x_tp / n)
    );

    constrained_fit_cmp(prec, rec, n3, n, p_h1);

    const double nll_h1 = (
        -2. * details::xlogy(x_tn, p_h1[0])
        - 2. * details::xlogy(x_fp, p_h1[1])
        - 2. * details::xlogy(x_fn, p_h1[2])
        - 2. * details::xlogy(x_tp, p_h1[3])
    );
    return nll_h1 - nll_h0;
}  // prof_loglike

inline void multn_uncertainty(
    const int64_t n_bins,
    const int64_t* __restrict conf_mat,
    double* result,
    double* bounds,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4
) {
    // -- memory allocation --
    // memory to be used by constrained_fit_cmp
    std::array<double, 4> probas;
    double* p = probas.data();
    // -- memory allocation --

    // obtain prec_start, prec_delta, rec_start, rec_delta are set
    details::get_pr_grid_delta(n_bins, conf_mat, bounds, n_sigmas, epsilon);
    double prec = bounds[0];
    const double prec_delta = bounds[1];
    const double rec_delta = bounds[4];
    auto recs_safe = std::unique_ptr<double[]>(
        details::linspace(bounds[3], bounds[5], n_bins, rec_delta)
    );
    double* recs = recs_safe.get();

    auto nll_store = prof_loglike_t();
    prof_loglike_t* nll_ptr = &nll_store;
    set_prof_loglike_store(conf_mat, nll_ptr);

    int64_t idx = 0;
    for (int64_t i = 0; i < n_bins; i++) {
        for (int64_t j = 0; j < n_bins; j++) {
            result[idx] = prof_loglike(prec, recs[j], nll_ptr, p);
            idx++;
        }
        prec += prec_delta;
    }
}  // multn_uncertainty

inline void multn_uncertainty_over_grid(
    const int64_t n_prec_bins,
    const int64_t n_rec_bins,
    const double* prec_grid,
    const double* rec_grid,
    const int64_t* __restrict conf_mat,
    double* scores,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4
) {

    // give scores a high enough initial value that the chi2 p-values will be close to zero
    std::fill(scores, scores + n_prec_bins * n_rec_bins, 1e4);
    // -- memory allocation --
    // memory to be used by constrained_fit_cmp
    std::array<double, 4> probas;
    double* p = probas.data();

    std::array<int64_t, 6> bounds;
    int64_t* idx_bounds = bounds.data();

    auto nll_store = prof_loglike_t();
    prof_loglike_t* nll_ptr = &nll_store;
    // -- memory allocation --

    // obtain the indexes over which to loop
    // sets prec_idx_min, prec_idx_max, rec_idx_min, rec_idx_max
    details::get_pr_grid_bounds(
        n_prec_bins, n_rec_bins, conf_mat,
        prec_grid, rec_grid, idx_bounds,
        n_sigmas, epsilon
    );
    const int64_t prec_idx_min = idx_bounds[0];
    const int64_t prec_idx_max = idx_bounds[1];
    const int64_t rec_idx_min = idx_bounds[2];
    const int64_t rec_idx_max = idx_bounds[3];

    set_prof_loglike_store(conf_mat, nll_ptr);

    double prec;
    double score;
    int64_t idx;
    int64_t odx;
    for (int64_t i = prec_idx_min; i < prec_idx_max; i++) {
        prec = prec_grid[i];
        odx = i * n_rec_bins;
        for (int64_t j = rec_idx_min; j < rec_idx_max; j++) {
            score = prof_loglike(prec, rec_grid[j], nll_ptr, p);
            idx = odx + j;
            // log likelihoods and thus always positive
            if (score < scores[idx]) {
                scores[idx] = score;
            }
        }
    }
}  // multn_uncertainty_over_grid

inline void multn_uncertainty_over_grid_thresholds(
    const int64_t n_prec_bins,
    const int64_t n_rec_bins,
    const int64_t n_conf_mats,
    const double* prec_grid,
    const double* rec_grid,
    const int64_t* __restrict conf_mat,
    double* scores,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4
) {
    // give scores a high enough initial value that the chi2 p-values will be close to zero
    std::fill(scores, scores + n_prec_bins * n_rec_bins, 1e4);

    // -- memory allocation --
    // memory to be used by constrained_fit_cmp
    std::array<double, 4> probas;
    double* p = probas.data();

    auto nll_store = prof_loglike_t();
    prof_loglike_t* nll_ptr = &nll_store;

    auto bounds = details::PrGridBounds(
        n_prec_bins,
        n_rec_bins,
        n_sigmas,
        epsilon,
        prec_grid,
        rec_grid
    );

    double prec;
    double score;
    int64_t idx;
    // -- memory allocation --

    for (int64_t k = 0; k < n_conf_mats; k++) {
        // update to new conf_mat
        set_prof_loglike_store(conf_mat, nll_ptr);
        bounds.compute_bounds(conf_mat);

        for (int64_t i = bounds.prec_idx_min; i < bounds.prec_idx_max; i++) {
            prec = prec_grid[i];
            for (int64_t j = bounds.rec_idx_min; j < bounds.rec_idx_max; j++) {
                score = prof_loglike(prec, rec_grid[j], nll_ptr, p);
                idx = (i * n_rec_bins) + j;
                if (score < scores[idx]) {
                    scores[idx] = score;
                }
            }
        }
        // increment ptr
        conf_mat += 4;
    }
}  // multn_uncertainty_over_grid_thresholds

#ifdef MMU_HAS_OPENMP_SUPPORT
inline void multn_uncertainty_over_grid_thresholds_mt(
    const int64_t n_prec_bins,
    const int64_t n_rec_bins,
    const int64_t n_conf_mats,
    const double* prec_grid,
    const double* rec_grid,
    const int64_t* __restrict conf_mat,
    double* scores,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4,
    const int64_t n_threads = 4
) {
    const int64_t n_elem = n_prec_bins * n_rec_bins;
    const int64_t t_elem = n_elem * n_threads;
    auto thread_scores = std::unique_ptr<double[]>(new double[t_elem]);

    std::fill(thread_scores.get(), thread_scores.get() + t_elem, 1e4);
#pragma omp parallel num_threads(n_threads) shared(n_prec_bins, n_rec_bins, n_conf_mats, prec_grid, rec_grid, conf_mat, n_sigmas, epsilon)
    {
        double* thread_block = thread_scores.get() + (omp_get_thread_num() * n_elem);

        // -- memory allocation --
        // memory to be used by constrained_fit_cmp
        std::array<double, 4> probas;
        double* p = probas.data();

        auto nll_store = prof_loglike_t();
        prof_loglike_t* nll_ptr = &nll_store;

        auto bounds = details::PrGridBounds(
            n_prec_bins,
            n_rec_bins,
            n_sigmas,
            epsilon,
            prec_grid,
            rec_grid
        );

        double prec;
        double score;
        int64_t idx;
        // -- memory allocation --
        const int64_t* lcm;

#pragma omp for
        for (int64_t k = 0; k < n_conf_mats; k++) {
            lcm = conf_mat + (k * 4);
            // update to new conf_mat
            set_prof_loglike_store(lcm, nll_ptr);
            bounds.compute_bounds(lcm);

            for (int64_t i = bounds.prec_idx_min; i < bounds.prec_idx_max; i++) {
                prec = prec_grid[i];
                for (int64_t j = bounds.rec_idx_min; j < bounds.rec_idx_max; j++) {
                    score = prof_loglike(prec, rec_grid[j], nll_ptr, p);
                    idx = (i * n_rec_bins) + j;
                    if (score < thread_block[idx]) {
                        thread_block[idx] = score;
                    }
                }
            }
        }
    } // omp parallel

    // collect the scores
    auto offsets = std::unique_ptr<int64_t[]>(new int64_t[n_threads]);
    for (int64_t j = 0; j < n_threads; j++) {
        offsets[j] = j * n_elem;
    }

    double tscore;
    double min_score;
    for (int64_t i = 0; i < n_elem; i++) {
        min_score = 1e4;
        for (int64_t j = 0; j < n_threads; j++) {
            tscore = thread_scores[i + offsets[j]];
            if (tscore < min_score) {
                min_score = tscore;
            }
        }
        scores[i] = min_score;
    }
}  // multn_uncertainty_over_grid_thresholds_mt
#endif  // MMU_HAS_OPENMP_SUPPORT

inline double prof_loglike_simulation_cov(
    const int64_t n_sims,
    random::pcg64_dxsm& rng,
    const double prec,
    const double rec,
    const double nll_obs,
    const int64_t n,
    const double* p,
    random::details::binomial_t* sptr,
    int64_t* mult_ptr,
    double* p_sim_ptr
) {
    int64_t checks = 0;
    for (int64_t i = 0; i < n_sims; i++) {
        // random_multinomial is not guarentee that all values are set
        zero_array<int64_t, 4>(mult_ptr);
        // draw a multinomial sample
        random::details::random_multinomial(rng, n, mult_ptr, p, 4, sptr);
        checks += prof_loglike(prec, rec, n, mult_ptr, p_sim_ptr) < nll_obs;
    }
    return static_cast<double>(checks) / n_sims;
} // prof_loglike_simulation_cov


inline void simulate_multn_uncertainty(
    const int64_t n_sims,
    const int64_t n_bins,
    const int64_t* __restrict conf_mat,
    double* scores,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4,
    const uint64_t seed = 0,
    const uint64_t stream = 0
) {
    random::pcg64_dxsm rng;
    if (seed == 0) {
        random::pcg_seed_seq seed_source;
        rng.seed(seed_source);
    } else if (stream != 0) {
        rng.seed(seed, stream);
    } else {
        rng.seed(seed);
    }

    // -- memory allocation --
    // memory to be used by constrained_fit_cmp
    std::array<double, 4> probas;
    double* p = probas.data();

    // allocate memory block to be used by multinomial
    std::array<int64_t, 4> mult;
    int64_t* mult_ptr = mult.data();

    // array used to store probabilities for the multinomial alternative
    std::array<double, 4> p_sim;
    double* p_sim_ptr = p_sim.data();

    // array used to store the bounds of loops
    std::array<double, 6> bounds;

    // struct used by the multinomial generation
    auto binom_store = random::details::binomial_t();
    random::details::binomial_t* sptr = &binom_store;

    // struct used to store the elements used in the computation
    // of the profile log-likelihood
    auto nll_store = prof_loglike_t();
    prof_loglike_t* nll_ptr = &nll_store;
    // -- memory allocation --

    // obtain prec_start, prec_delta, rec_start, rec_delta are set
    details::get_pr_grid_delta(n_bins, conf_mat, bounds.data(), n_sigmas, epsilon);
    double prec = bounds[0];
    const double prec_delta = bounds[1];
    const double rec_delta = bounds[4];
    auto recs_safe = std::unique_ptr<double[]>(
        details::linspace(bounds[3], bounds[5], n_bins, rec_delta)
    );
    double* recs = recs_safe.get();

    set_prof_loglike_store(conf_mat, nll_ptr);
    const int64_t n = nll_ptr->in;

    double rec;
    double nll_obs;
    int64_t idx = 0;
    for (int64_t i = 0; i < n_bins; i++) {
        for (int64_t j = 0; j < n_bins; j++) {
            // prof_loglike also sets p which we can re-use in prof_loglike sim
            rec = recs[j];
            nll_obs = prof_loglike(prec, rec, nll_ptr, p);
            scores[idx] = prof_loglike_simulation_cov(
                n_sims, rng, prec, rec, nll_obs, n, p, sptr, mult_ptr, p_sim_ptr
            );
            idx++;
        }
        prec += prec_delta;
    }
} // simulate_multn_uncertainty

#ifdef MMU_HAS_OPENMP_SUPPORT
inline void simulate_multn_uncertainty_mt(
    const int64_t n_sims,
    const int64_t n_bins,
    const int64_t* __restrict conf_mat,
    double* scores,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4,
    const uint64_t seed = 0,
    const uint64_t stream = 0
) {
    random::pcg64_dxsm rng;
    if (seed == 0) {
        random::pcg_seed_seq seed_source;
        rng.seed(seed_source);
    } else if (stream != 0) {
        rng.seed(seed, stream);
    } else {
        rng.seed(seed);
    }

    // -- memory allocation --
    // memory to be used by constrained_fit_cmp
    std::array<double, 4> probas;
    double* p = probas.data();

    // allocate memory block to be used by multinomial
    std::array<int64_t, 4> mult;
    int64_t* mult_ptr = mult.data();

    // array used to store probabilities for the multinomial alternative
    std::array<double, 4> p_sim;
    double* p_sim_ptr = p_sim.data();

    // array used to store the bounds of loops
    std::array<double, 6> bounds;

    // struct used by the multinomial generation
    auto binom_store = random::details::binomial_t();
    random::details::binomial_t* sptr = &binom_store;

    // struct used to store the elements used in the computation
    // of the profile log-likelihood
    auto nll_store = prof_loglike_t();
    prof_loglike_t* nll_ptr = &nll_store;
    // -- memory allocation --

    // obtain prec_start, prec_delta, rec_start, rec_delta are set
    details::get_pr_grid_delta(n_bins, conf_mat, bounds.data(), n_sigmas, epsilon);
    double prec = bounds[0];
    const double prec_delta = bounds[1];
    const double rec_delta = bounds[4];
    auto recs_safe = std::unique_ptr<double[]>(
        details::linspace(bounds[3], bounds[5], n_bins, rec_delta)
    );
    double* recs = recs_safe.get();

    set_prof_loglike_store(conf_mat, nll_ptr);
    const int64_t n = nll_ptr->in;

    double rec;
    double nll_obs;
    int64_t idx = 0;
    for (int64_t i = 0; i < n_bins; i++) {
        for (int64_t j = 0; j < n_bins; j++) {
            // prof_loglike also sets p which we can re-use in prof_loglike sim
            rec = recs[j];
            nll_obs = prof_loglike(prec, rec, nll_ptr, p);
            scores[idx] = prof_loglike_simulation_cov(
                n_sims, rng, prec, rec, nll_obs, n, p, sptr, mult_ptr, p_sim_ptr
            );
            idx++;
        }
        prec += prec_delta;
    }
} // simulate_multn_uncertainty_mt
#endif  // MMU_HAS_OPENMP_SUPPORT

}  // namespace core
}  // namespace mmu

#endif  // INCLUDE_MMU_CORE_MULTN_LOGLIKE_HPP_
