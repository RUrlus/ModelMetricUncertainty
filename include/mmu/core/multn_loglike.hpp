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
    // -- memory allocation --

    // obtain prec_start, prec_delta, rec_start, rec_delta are set
    details::get_pr_grid_delta(n_bins, conf_mat, bounds, n_sigmas, epsilon);
    double prec = bounds[0];
    const double prec_delta = bounds[1];
    const double rec_delta = bounds[4];
    auto recs_safe = std::unique_ptr<double[]>(details::linspace(bounds[3], bounds[5], n_bins, rec_delta));
    double* recs = recs_safe.get();

    auto nll_store = prof_loglike_t();
    prof_loglike_t* nll_ptr = &nll_store;
    set_prof_loglike_store(conf_mat, nll_ptr);

    int64_t idx = 0;
    for (int64_t i = 0; i < n_bins; i++) {
        for (int64_t j = 0; j < n_bins; j++) {
            // prof_loglike also sets p which we can re-use in prof_loglike sim
            result[idx] = prof_loglike(prec, recs[j], nll_ptr, p);
            idx++;
        }
        prec += prec_delta;
    }
}  // multn_uncertainty

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
    double* result,
    double* bounds,
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
    // -- memory allocation --

    // obtain prec_start, prec_delta, rec_start, rec_delta are set
    details::get_pr_grid_delta(n_bins, conf_mat, bounds, n_sigmas, epsilon);
    double prec = bounds[0];
    const double prec_delta = bounds[1];
    const double rec_delta = bounds[4];
    auto recs_safe = std::unique_ptr<double[]>(details::linspace(bounds[3], bounds[5], n_bins, rec_delta));
    double* recs = recs_safe.get();

    auto nll_store = prof_loglike_t();
    prof_loglike_t* nll_ptr = &nll_store;
    set_prof_loglike_store(conf_mat, nll_ptr);
    const int64_t n = nll_ptr->in;

    auto binom_store = random::details::binomial_t();
    random::details::binomial_t* sptr = &binom_store;

    // allocate memory block to be used by multinomial
    std::array<int64_t, 4> mult;
    int64_t* mult_ptr = mult.data();

    std::array<double, 4> p_sim;
    double* p_sim_ptr = p_sim.data();

    double rec;
    double nll_obs;
    int64_t idx = 0;
    for (int64_t i = 0; i < n_bins; i++) {
        for (int64_t j = 0; j < n_bins; j++) {
            // prof_loglike also sets p which we can re-use in prof_loglike sim
            rec = recs[j];
            nll_obs = prof_loglike(prec, rec, nll_ptr, p);
            result[idx] = prof_loglike_simulation_cov(
                n_sims, rng, prec, rec, nll_obs, n, p, sptr, mult_ptr, p_sim_ptr
            );
            idx++;
        }
        prec += prec_delta;
    }
} // simulate_multn_uncertainty

}  // namespace core
}  // namespace mmu

#endif  // INCLUDE_MMU_CORE_MULTN_LOGLIKE_HPP_
