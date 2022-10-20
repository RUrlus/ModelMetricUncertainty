/* roc_multn_loglike.hpp -- Implementation Multinomial uncertainty over
 * TPR-FPR using Wilk's theorem Copyright 2022 Ralph Urlus
 */
#ifndef INCLUDE_MMU_CORE_ROC_MULTN_LOGLIKE_HPP_
#define INCLUDE_MMU_CORE_ROC_MULTN_LOGLIKE_HPP_

#if defined(MMU_HAS_OPENMP_SUPPORT)
#include <omp.h>
#endif  // MMU_HAS_OPENMP_SUPPORT

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>

#include <mmu/core/bvn_error.hpp>
#include <mmu/core/common.hpp>
#include <mmu/core/grid_bounds.hpp>
#include <mmu/core/metrics.hpp>
#include <mmu/core/random.hpp>

/* conf_mat layout:
 *  0 TN
 *  1 FP
 *  2 FN
 *  3 TP
 */

namespace mmu {
namespace core {
namespace roc {
/* Compute the most conservative probabilities for a given confusion matrix
 * constrained by TPR and FPR. For single function call.
 *
 * Parameters
 * ----------
 *  tpr : True Positive Rate
 *  fpr : False Positive Rate
 *  conf_mat : confusion matrix with order TN, FP, FN, TP
 *  probas : result array
 */
inline void constrained_fit_cmp(
    const double tpr,
    const double fpr,
    const int64_t* __restrict conf_mat,
    double* __restrict probas) {
    // n2 = FN + TP
    const int64_t in2 = conf_mat[2] + conf_mat[3];
    const auto n4 = static_cast<double>(conf_mat[0] + conf_mat[1] + in2);
    const auto n2 = static_cast<double>(in2);

    // guard against divide by zero
    constexpr double ratio_fill = (1.0 - 1e-12) / 1e-12;
    const double tpr_ratio = tpr > std::numeric_limits<double>::epsilon()
                                 ? (1.0 - tpr) / tpr
                                 : ratio_fill;
    constexpr double inv_fill = 1.0 / 1e-12;
    const double tpr_inv = tpr > std::numeric_limits<double>::epsilon()
                                  ? 1.0 / tpr
                                  : inv_fill;
    const double p_tp = (tpr * n2) / n4;
    const double p_fn = tpr_ratio * p_tp;
    const double p_fp = (fpr*(tpr-p_tp)) * tpr_inv;
    // guard against floating point noise resulting in negative probabilities
    const double p_tn = std::max(1. - p_fn - p_fp - p_tp, 0.0);
    probas[0] = p_tn;
    probas[1] = p_fp;
    probas[2] = p_fn;
    probas[3] = p_tp;
}  // constrained_fit_cmp

/* Compute the most conservative probabilities for a given confusion matrix
 * constrained by TPR and FPR. Called in loops.
 *
 * Parameters
 * ----------
 *  tpr : True Positive Rate
 *  fpr : False Positive Rate
 *  conf_mat : confusion matrix with order TN, FP, FN, TP
 *  probas : result array
 */
inline void constrained_fit_cmp(
    const double tpr,
    const double fpr,
    const double n2, // Remark this parameter is n3 in precision-recall
    const double n4,
    double* __restrict probas) {
    const double tpr_ratio = (1.0 - tpr) / tpr;
    const double tpr_inv = 1.0 / tpr;
    probas[3] = (tpr * n2) / n4;
    probas[2] = tpr_ratio * probas[3];
    probas[1] = (fpr*(tpr-probas[3])) * tpr_inv;
    // guard against floating point noise resulting in negative probabilities
    probas[0] = std::max(1. - probas[1] - probas[2] - probas[3], 0.0);
}  // constrained_fit_cmp

typedef struct s_prof_loglike_t {
    int64_t in;
    double n2;
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
    prof_loglike_t* store) {
    // total number of entries in the confusion matrix
    const int64_t in2 = conf_mat[2] + conf_mat[3];
    // Remark n2 (n3 for precision-recall) is set here:
    store->n2 = static_cast<double>(in2);
    store->in = conf_mat[0] + conf_mat[1] + in2;
    store->n = static_cast<double>(store->in);

    store->x_tn = static_cast<double>(conf_mat[0]);
    store->x_fp = static_cast<double>(conf_mat[1]);
    store->x_fn = static_cast<double>(conf_mat[2]);
    store->x_tp = static_cast<double>(conf_mat[3]);

    store->p_tn = store->x_tn / store->n;
    store->p_fp = store->x_fp / store->n;
    store->p_fn = store->x_fn / store->n;
    store->p_tp = store->x_tp / store->n;

    store->nll_h0 = -2
                    * (mmu::core::details::xlogy(store->x_tn, store->p_tn)
                       + details::xlogy(store->x_fp, store->p_fp)
                       + details::xlogy(store->x_fn, store->p_fn)
                       + details::xlogy(store->x_tp, store->p_tp));
}

/* Compute -2logp of multinomial distribution given a Y and X.
 * Called in loops.
 *
 * Step 1: fit with all parameters free
 * Step 2: fit multinomial with fixed Y and X
 * Step 3: compute -2ln(L_h0 / L_h1)
 *
 * We compute -2ln(L_h0) + 2ln(L_h1))
 */
// TODO: this function is exactly the same, it just need to call a different constrained_fit_cmp()
inline double prof_loglike(
    const double prec,
    const double rec,
    prof_loglike_t* store,
    double* __restrict p_h0) {
    constrained_fit_cmp(prec, rec, store->n2, store->n, p_h0);
    const double nll_h1 = -2
                          * (details::xlogy(store->x_tn, p_h0[0])
                             + details::xlogy(store->x_fp, p_h0[1])
                             + details::xlogy(store->x_fn, p_h0[2])
                             + details::xlogy(store->x_tp, p_h0[3]));
    return nll_h1 - store->nll_h0;
}  // prof_loglike

/* Compute -2logp of multinomial distribution given a TPR and FPR.
 * For single function call.
 *
 * Step 1: fit with all parameters free
 * Step 2: fit multinomial with fixed TPR and FPR
 * Step 3: compute -2ln(L_h0 / L_h1)
 *
 * We compute -2ln(L_h0) + 2ln(L_h1))
 */
// TODO argument where not renamed
inline double prof_loglike(
    const double prec,
    const double rec,
    const double n,
    const int64_t* __restrict conf_mat,
    double* __restrict p_h0) {
    const auto n2 = static_cast<double>(conf_mat[2] + conf_mat[3]); // note that n2 is computed here
    const auto x_tn = static_cast<double>(conf_mat[0]);
    const auto x_fp = static_cast<double>(conf_mat[1]);
    const auto x_fn = static_cast<double>(conf_mat[2]);
    const auto x_tp = static_cast<double>(conf_mat[3]);

    // constitutes the optimal/unconstrained fit of a multinomial
    const double nll_h0
        = -2
          * (details::xlogy(x_tn, x_tn / n) + details::xlogy(x_fp, x_fp / n)
             + details::xlogy(x_fn, x_fn / n) + details::xlogy(x_tp, x_tp / n));

    constrained_fit_cmp(prec, rec, n2, n, p_h0);

    const double nll_h1
        = -2.
          * (details::xlogy(x_tn, p_h0[0]) + details::xlogy(x_fp, p_h0[1])
             + details::xlogy(x_fn, p_h0[2]) + details::xlogy(x_tp, p_h0[3]));
    return nll_h1 - nll_h0;
}  // prof_loglike

// TODO: to refactor, it wasn't changed
inline double multn_chi2_score(
    const double prec,
    const double rec,
    const int64_t* __restrict conf_mat,
    const double epsilon = 1e-4) {
    // -- memory allocation --
    // memory to be used by constrained_fit_cmp
    std::array<double, 4> probas;
    double* p = probas.data();
    // -- memory allocation --
    const auto n = static_cast<double>(
        conf_mat[0] + conf_mat[1] + conf_mat[2] + conf_mat[3]);
    const double max_val = 1.0 - epsilon;
    return prof_loglike(
        details::clamp(prec, epsilon, max_val),
        details::clamp(rec, epsilon, max_val),
        n,
        conf_mat,
        p);
}  // multn_chi2_score

// TODO: to refactor, it wasn't changed
inline void multn_chi2_scores(
    const int64_t n_points,
    const double* precs,
    const double* recs,
    const int64_t* __restrict conf_mat,
    double* scores,
    const double epsilon = 1e-4) {
    // -- memory allocation --
    // memory to be used by constrained_fit_cmp
    std::array<double, 4> probas;
    double* p = probas.data();

    auto nll_store = prof_loglike_t();
    prof_loglike_t* nll_ptr = &nll_store;
    set_prof_loglike_store(conf_mat, nll_ptr);
    // -- memory allocation --

    const double max_val = 1.0 - epsilon;
    for (int64_t i = 0; i < n_points; ++i) {
        scores[i] = prof_loglike(
            details::clamp(precs[i], epsilon, max_val),
            details::clamp(recs[i], epsilon, max_val),
            nll_ptr,
            p);
    }
}  // multn_chi2_scores

#ifdef MMU_HAS_OPENMP_SUPPORT
// TODO: to refactor, it wasn't changed
inline void multn_chi2_scores_mt(
    const int64_t n_points,
    const double* precs,
    const double* recs,
    const int64_t* __restrict conf_mat,
    double* scores,
    const double epsilon = 1e-4) {
    // -- memory allocation --
    // memory to be used by constrained_fit_cmp
    std::array<double, 4> probas;
    double* p = probas.data();
    auto nll_store = prof_loglike_t();
    prof_loglike_t* nll_ptr = &nll_store;
    set_prof_loglike_store(conf_mat, nll_ptr);
    // -- memory allocation --
    const double max_val = 1.0 - epsilon;

#pragma omp parallel shared(precs, recs, nll_ptr, scores)
    {
#pragma omp for
        for (int64_t i = 0; i < n_points; ++i) {
            scores[i] = prof_loglike(
                details::clamp(precs[i], epsilon, max_val),
                details::clamp(recs[i], epsilon, max_val),
                nll_ptr,
                p);
        }
    }  // omp parallel
}  // multn_chi2_scores_mt
#endif  // MMU_HAS_OPENMP_SUPPORT

// TODO: to refactor, it wasn't changed
inline void multn_error(
    const int64_t n_bins,
    const int64_t* __restrict conf_mat,
    double* __restrict result,
    double* __restrict bounds,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4) {
    // -- memory allocation --
    // memory to be used by constrained_fit_cmp
    std::array<double, 4> probas;
    double* p = probas.data();
    // -- memory allocation --

    // obtain prec_start, prec_end, rec_start, rec_end
    get_grid_bounds(conf_mat, bounds, n_sigmas, epsilon);
    auto rec_grid = std::unique_ptr<double[]>(new double[n_bins]);
    details::linspace(bounds[2], bounds[3], n_bins, rec_grid.get());
    const double prec_start = bounds[0];
    const double prec_delta
        = (bounds[1] - bounds[0]) / static_cast<double>(n_bins - 1);

    auto nll_store = prof_loglike_t();
    prof_loglike_t* nll_ptr = &nll_store;
    set_prof_loglike_store(conf_mat, nll_ptr);

    int64_t idx = 0;
    for (int64_t i = 0; i < n_bins; i++) {
        double prec = prec_start + (static_cast<double>(i) * prec_delta);
        for (int64_t j = 0; j < n_bins; j++) {
            result[idx] = prof_loglike(prec, rec_grid[j], nll_ptr, p);
            idx++;
        }
    }
}  // multn_error

#ifdef MMU_HAS_OPENMP_SUPPORT
// TODO: to refactor, it wasn't changed
inline void multn_error_mt(
    const int64_t n_bins,
    const int64_t* __restrict conf_mat,
    double* __restrict result,
    double* __restrict bounds,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4,
    const int n_threads = 4) {
    // obtain prec_start, prec_end, rec_start, rec_end
    get_grid_bounds(conf_mat, bounds, n_sigmas, epsilon);
    auto rec_grid = std::unique_ptr<double[]>(new double[n_bins]);
    details::linspace(bounds[2], bounds[3], n_bins, rec_grid.get());
    const double prec_start = bounds[0];
    const double prec_delta
        = (bounds[1] - bounds[0]) / static_cast<double>(n_bins - 1);

    auto nll_store = prof_loglike_t();
    prof_loglike_t* nll_ptr = &nll_store;
    set_prof_loglike_store(conf_mat, nll_ptr);
#pragma omp parallel num_threads(n_threads) shared(rec_grid, nll_ptr, result)
    {
        // -- memory allocation --
        // memory to be used by constrained_fit_cmp
        std::array<double, 4> probas;
        double* p = probas.data();
        // -- memory allocation --

#pragma omp for
        for (int64_t i = 0; i < n_bins; i++) {
            int64_t idx;
            double prec;
            prec = prec_start + (static_cast<double>(i) * prec_delta);
            idx = i * n_bins;
            for (int64_t j = 0; j < n_bins; j++) {
                result[idx + j] = prof_loglike(prec, rec_grid[j], nll_ptr, p);
            }
        }
    }  // omp parallel
}  // multn_error_mt
#endif  // MMU_HAS_OPENMP_SUPPORT

// TODO: to refactor, it wasn't changed
inline void multn_grid_error(
    const int64_t n_prec_bins,
    const int64_t n_rec_bins,
    const double* __restrict prec_grid,
    const double* __restrict rec_grid,
    const int64_t* __restrict conf_mat,
    double* __restrict scores,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4) {
    // give scores a high enough initial value that the chi2 p-values will be
    // close to zero i.e. ppf(1-1e-14) --> 64.47398179869367
    std::fill(
        scores, scores + n_prec_bins * n_rec_bins, MULT_DEFAULT_CHI2_SCORE);
    // -- memory allocation --
    // memory to be used by constrained_fit_cmp
    std::array<double, 4> probas;
    double* p = probas.data();

    std::array<int64_t, 3> bounds;
    int64_t* idx_bounds = bounds.data();

    auto nll_store = prof_loglike_t();
    prof_loglike_t* nll_ptr = &nll_store;
    // -- memory allocation --

    // obtain the indexes over which to loop
    // sets prec_idx_min, prec_idx_max, rec_idx_min, rec_idx_max
    get_grid_bounds(
        n_prec_bins,
        n_rec_bins,
        conf_mat,
        prec_grid,
        rec_grid,
        idx_bounds,
        n_sigmas,
        epsilon);
    const int64_t prec_idx_min = idx_bounds[0];
    const int64_t prec_idx_max = idx_bounds[1];
    const int64_t rec_idx_min = idx_bounds[2];
    const int64_t rec_idx_max = idx_bounds[3];

    set_prof_loglike_store(conf_mat, nll_ptr);

    double score;
    int64_t idx;
    for (int64_t i = prec_idx_min; i < prec_idx_max; i++) {
        double prec = prec_grid[i];
        int64_t odx = i * n_rec_bins;
        for (int64_t j = rec_idx_min; j < rec_idx_max; j++) {
            score = prof_loglike(prec, rec_grid[j], nll_ptr, p);
            idx = odx + j;
            // log likelihoods and thus always positive
            if (score < scores[idx]) {
                scores[idx] = score;
            }
        }
    }
}  // multn_grid_error

// TODO: to refactor, it wasn't changed
inline void multn_grid_curve_error(
    const int64_t n_prec_bins,
    const int64_t n_rec_bins,
    const int64_t n_conf_mats,
    const double* __restrict prec_grid,
    const double* __restrict rec_grid,
    const int64_t* __restrict conf_mat,
    double* __restrict scores,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4) {
    // give scores a high enough initial value that the chi2 p-values will be
    // close to zero i.e. ppf(1-1e-14) --> 64.47398179869367
    std::fill(
        scores, scores + n_prec_bins * n_rec_bins, MULT_DEFAULT_CHI2_SCORE);

    // -- memory allocation --
    // memory to be used by constrained_fit_cmp
    std::array<double, 4> probas;
    double* p = probas.data();

    auto nll_store = prof_loglike_t();
    prof_loglike_t* nll_ptr = &nll_store;

    auto bounds = GridBounds(
        n_prec_bins, n_rec_bins, n_sigmas, epsilon, prec_grid, rec_grid);

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
}  // multn_grid_curve_error

#ifdef MMU_HAS_OPENMP_SUPPORT
// TODO: to refactor, it wasn't changed
inline void multn_grid_curve_error_mt(
    const int64_t n_prec_bins,
    const int64_t n_rec_bins,
    const int64_t n_conf_mats,
    const double* __restrict prec_grid,
    const double* __restrict rec_grid,
    const int64_t* __restrict conf_mat,
    double* __restrict scores,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4,
    const int64_t n_threads = 4) {
    const int64_t n_elem = n_prec_bins * n_rec_bins;
    const int64_t t_elem = n_elem * n_threads;
    auto thread_scores = std::unique_ptr<double[]>(new double[t_elem]);

    // give scores a high enough initial value that the chi2 p-values will be
    // close to zero i.e. ppf(1-1e-14) --> 64.47398179869367
    std::fill(
        thread_scores.get(),
        thread_scores.get() + t_elem,
        MULT_DEFAULT_CHI2_SCORE);
#pragma omp parallel num_threads(n_threads) shared( \
    n_prec_bins,                                    \
    n_rec_bins,                                     \
    n_conf_mats,                                    \
    prec_grid,                                      \
    rec_grid,                                       \
    conf_mat,                                       \
    n_sigmas,                                       \
    epsilon)
    {
        double* thread_block
            = thread_scores.get() + (omp_get_thread_num() * n_elem);

        // -- memory allocation --
        // memory to be used by constrained_fit_cmp
        std::array<double, 4> probas;
        double* p = probas.data();

        auto nll_store = prof_loglike_t();
        prof_loglike_t* nll_ptr = &nll_store;

        auto bounds = GridBounds(
            n_prec_bins, n_rec_bins, n_sigmas, epsilon, prec_grid, rec_grid);

        // -- memory allocation --

#pragma omp for
        for (int64_t k = 0; k < n_conf_mats; k++) {
            const int64_t* lcm = conf_mat + (k * 4);
            // update to new conf_mat
            set_prof_loglike_store(lcm, nll_ptr);
            bounds.compute_bounds(lcm);

            for (int64_t i = bounds.prec_idx_min; i < bounds.prec_idx_max;
                 i++) {
                double prec = prec_grid[i];
                int64_t odx = i * n_rec_bins;
                for (int64_t j = bounds.rec_idx_min; j < bounds.rec_idx_max;
                     j++) {
                    double score = prof_loglike(prec, rec_grid[j], nll_ptr, p);
                    int64_t idx = odx + j;
                    if (score < thread_block[idx]) {
                        thread_block[idx] = score;
                    }
                }
            }
        }
    }  // omp parallel

    // collect the scores
    auto offsets = std::unique_ptr<int64_t[]>(new int64_t[n_threads]);
    for (int64_t j = 0; j < n_threads; j++) {
        offsets[j] = j * n_elem;
    }

    for (int64_t i = 0; i < n_elem; i++) {
        // give scores a high enough initial value that the chi2 p-values will
        // be close to zero i.e. ppf(1-1e-14) --> 64.47398179869367
        double min_score = MULT_DEFAULT_CHI2_SCORE;
        for (int64_t j = 0; j < n_threads; j++) {
            double tscore = thread_scores[i + offsets[j]];
            if (tscore < min_score) {
                min_score = tscore;
            }
        }
        scores[i] = min_score;
    }
}  // multn_grid_curve_error_mt
#endif  // MMU_HAS_OPENMP_SUPPORT

struct simulation_store {
    random::details::binomial_t binom_store;
    random::pcg64_dxsm rng;
    // memory to be used by constrained_fit_cmp
    std::array<double, 4> probas;
    // memory to be used to store probabilities for the multinomial alternative
    std::array<double, 4> p_sim;
    // allocate memory block to be used by multinomial
    std::array<int64_t, 4> mult;
    const int64_t n_sims;
    int64_t n;
    random::details::binomial_t* sptr;
    double* p;
    double* p_sim_ptr;
    int64_t* mult_ptr;

    simulation_store(
        random::pcg64_dxsm rng,
        const int64_t n_sims,
        const int64_t n)
        : binom_store{random::details::binomial_t()},
          rng{rng},
          n_sims{n_sims},
          n{n},
          sptr{&binom_store},
          p(probas.data()),
          p_sim_ptr(p_sim.data()),
          mult_ptr(mult.data()) {}
};

// TODO: to refactor, it wasn't changed
inline double prof_loglike_simulation_cov(
    const double prec,
    const double rec,
    const double nll_obs,
    simulation_store& sim_store) {
    int64_t checks = 0;
    for (int64_t i = 0; i < sim_store.n_sims; i++) {
        // random_multinomial does not guarentee that all values are set
        zero_array<int64_t, 4>(sim_store.mult_ptr);
        // draw a multinomial sample
        random::details::random_multinomial(
            sim_store.rng,
            sim_store.n,
            sim_store.mult_ptr,
            sim_store.p,
            4,
            sim_store.sptr);
        checks += prof_loglike(
                      prec,
                      rec,
                      sim_store.n,
                      sim_store.mult_ptr,
                      sim_store.p_sim_ptr)
                  < nll_obs;
    }
    return static_cast<double>(checks) / sim_store.n_sims;
}  // prof_loglike_simulation_cov

// TODO: to refactor, it wasn't changed
inline void multn_sim_error(
    const int64_t n_sims,
    const int64_t n_bins,
    const int64_t* __restrict conf_mat,
    double* scores,
    double* bounds,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4,
    const uint64_t seed = 0,
    const uint64_t stream = 0) {
    random::pcg64_dxsm rng;
    if (seed == 0) {
        random::pcg_seed_seq seed_source;
        rng.seed(seed_source);
    } else if (stream != 0) {
        rng.seed(seed, stream);
    } else {
        rng.seed(seed);
    }

    // obtain prec_start, prec_end, rec_start, rec_end
    get_grid_bounds(conf_mat, bounds, n_sigmas, epsilon);
    auto rec_grid = std::unique_ptr<double[]>(new double[n_bins]);
    details::linspace(bounds[2], bounds[3], n_bins, rec_grid.get());
    const double prec_start = bounds[0];
    const double prec_delta
        = (bounds[1] - bounds[0]) / static_cast<double>(n_bins - 1);

    // struct used to store the elements used in the computation
    // of the profile log-likelihood
    auto nll_store = prof_loglike_t();
    prof_loglike_t* nll_ptr = &nll_store;
    set_prof_loglike_store(conf_mat, nll_ptr);
    const int64_t n = nll_ptr->in;

    // struct with parameters used in the simulations
    auto sim_store = simulation_store(rng, n_sims, n);

    // give scores a high enough initial value that the chi2 p-values will be
    // close to zero i.e. ppf(1-1e-14) --> 64.47398179869367
    std::fill(scores, scores + n_bins * n_bins, MULT_DEFAULT_CHI2_SCORE);

    double rec;
    double prec;
    double nll_obs;
    int64_t idx = 0;
    for (int64_t i = 0; i < n_bins; i++) {
        prec = prec_start + (i * prec_delta);
        for (int64_t j = 0; j < n_bins; j++) {
            // prof_loglike also sets p which we can re-use in prof_loglike sim
            rec = rec_grid[j];
            nll_obs = prof_loglike(prec, rec, nll_ptr, sim_store.p);
            scores[idx]
                = prof_loglike_simulation_cov(prec, rec, nll_obs, sim_store);
            idx++;
        }
    }
}  // multn_sim_error

#ifdef MMU_HAS_OPENMP_SUPPORT
// TODO: to refactor, it wasn't changed
inline void multn_sim_error_mt(
    const int64_t n_sims,
    const int64_t n_bins,
    const int64_t* __restrict conf_mat,
    double* __restrict scores,
    double* __restrict bounds,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4,
    const uint64_t seed = 0,
    const int n_threads = 4) {
    // obtain prec_start, prec_end, rec_start, rec_end
    get_grid_bounds(conf_mat, bounds, n_sigmas, epsilon);
    auto rec_grid = std::unique_ptr<double[]>(new double[n_bins]);
    details::linspace(bounds[2], bounds[3], n_bins, rec_grid.get());
    const double prec_start = bounds[0];
    const double prec_delta
        = (bounds[1] - bounds[0]) / static_cast<double>(n_bins - 1);

    random::pcg_seed_seq seed_source;
    std::array<uint64_t, 2> gen_seeds;
    seed_source.generate(gen_seeds.begin(), gen_seeds.end());

    // give scores a high enough initial value that the chi2 p-values will be
    // close to zero i.e. ppf(1-1e-14) --> 64.47398179869367
    std::fill(scores, scores + n_bins * n_bins, MULT_DEFAULT_CHI2_SCORE);

#pragma omp parallel num_threads(n_threads) \
    shared(conf_mat, gen_seeds, seed, prec_start, prec_delta, rec_grid)
    {
        random::pcg64_dxsm rng;
        if (seed == 0) {
            rng.seed(gen_seeds[0], omp_get_thread_num());
        } else {
            rng.seed(seed, omp_get_thread_num());
        }

        // struct used to store the elements used in the computation
        // of the profile log-likelihood
        auto nll_store = prof_loglike_t();
        prof_loglike_t* nll_ptr = &nll_store;
        set_prof_loglike_store(conf_mat, nll_ptr);
        const int64_t n = nll_ptr->in;

        // struct with parameters used in the simulations
        auto sim_store = simulation_store(rng, n_sims, n);

#pragma omp for
        for (int64_t i = 0; i < n_bins; i++) {
            double prec = prec_start + (static_cast<double>(i) * prec_delta);
            int64_t odx = i * n_bins;
            for (int64_t j = 0; j < n_bins; j++) {
                // prof_loglike also sets p which we can re-use in prof_loglike
                // sim
                double rec = rec_grid[j];
                double nll_obs = prof_loglike(prec, rec, nll_ptr, sim_store.p);
                scores[odx + j] = prof_loglike_simulation_cov(
                    prec, rec, nll_obs, sim_store);
            }
        }
    }  // omp parallel
}  // multn_sim_error_mt
#endif  // MMU_HAS_OPENMP_SUPPORT

#ifdef MMU_HAS_OPENMP_SUPPORT
// TODO: to refactor, it wasn't changed
inline void multn_sim_curve_error_mt(
    const int64_t n_sims,
    const int64_t n_prec_bins,
    const int64_t n_rec_bins,
    const int64_t n_conf_mats,
    const double* __restrict prec_grid,
    const double* __restrict rec_grid,
    const int64_t* __restrict conf_mat,
    double* __restrict scores,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4,
    const uint64_t seed = 0,
    const int64_t n_threads = 4) {
    const int64_t n_elem = n_prec_bins * n_rec_bins;
    const int64_t t_elem = n_elem * n_threads;
    auto thread_scores = std::unique_ptr<double[]>(new double[t_elem]);

    // give scores a high enough initial value that the chi2 p-values will be
    // close to zero i.e. ppf(1-1e-14) --> 64.47398179869367
    std::fill(thread_scores.get(), thread_scores.get() + t_elem, 1.0);

    random::pcg_seed_seq seed_source;
    std::array<uint64_t, 2> gen_seeds;
    seed_source.generate(gen_seeds.begin(), gen_seeds.end());
#pragma omp parallel num_threads(n_threads) shared( \
    n_prec_bins,                                    \
    n_rec_bins,                                     \
    n_conf_mats,                                    \
    prec_grid,                                      \
    rec_grid,                                       \
    conf_mat,                                       \
    n_sigmas,                                       \
    epsilon)
    {
        double* thread_block
            = thread_scores.get() + (omp_get_thread_num() * n_elem);

        // -- memory allocation --
        auto nll_store = prof_loglike_t();
        prof_loglike_t* nll_ptr = &nll_store;

        // struct with parameters used in the simulations
        //
        random::pcg64_dxsm rng;
        if (seed == 0) {
            rng.seed(gen_seeds[0], omp_get_thread_num());
        } else {
            rng.seed(seed, omp_get_thread_num());
        }
        auto sim_store = simulation_store(rng, n_sims, 0);

        auto bounds = GridBounds(
            n_prec_bins, n_rec_bins, n_sigmas, epsilon, prec_grid, rec_grid);

        double rec;
        double prec;
        double nll_obs;
        double score;
        int64_t idx;
        int64_t odx;
        // -- memory allocation --

#pragma omp for
        for (int64_t k = 0; k < n_conf_mats; k++) {
            const int64_t* lcm = conf_mat + (k * 4);
            // update to new conf_mat
            set_prof_loglike_store(lcm, nll_ptr);
            bounds.compute_bounds(lcm);
            sim_store.n = nll_store.in;

            for (int64_t i = bounds.prec_idx_min; i < bounds.prec_idx_max;
                 i++) {
                prec = prec_grid[i];
                odx = i * n_rec_bins;
                for (int64_t j = bounds.rec_idx_min; j < bounds.rec_idx_max;
                     j++) {
                    rec = rec_grid[j];
                    nll_obs = prof_loglike(prec, rec, nll_ptr, sim_store.p);
                    score = prof_loglike_simulation_cov(
                        prec, rec, nll_obs, sim_store);
                    idx = odx + j;
                    if (score < thread_block[idx]) {
                        thread_block[idx] = score;
                    }
                }
            }
        }
    }  // omp parallel

    // collect the scores
    auto offsets = std::unique_ptr<int64_t[]>(new int64_t[n_threads]);
    for (int64_t j = 0; j < n_threads; j++) {
        offsets[j] = j * n_elem;
    }

    for (int64_t i = 0; i < n_elem; i++) {
        double min_score = 1.0;
        for (int64_t j = 0; j < n_threads; j++) {
            double tscore = thread_scores[i + offsets[j]];
            if (tscore < min_score) {
                min_score = tscore;
            }
        }
        scores[i] = min_score;
    }
}  // multn_sim_curve_error_mt
#endif  // MMU_HAS_OPENMP_SUPPORT

}  // namespace roc
}  // namespace core
}  // namespace mmu

#endif  // INCLUDE_MMU_CORE_ROC_MULTN_LOGLIKE_HPP_
