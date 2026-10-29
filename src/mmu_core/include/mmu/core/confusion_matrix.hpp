/* confusion_matrix.hpp -- Implementation of binary classification confusion
 * matrix Copyright 2026 Ralph Urlus
 */
#pragma once

#include <algorithm>  // for sort, lower_bound
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <mmu/core/common.hpp>

/*                  pred
 *                0     1
 *  actual  0    TN    FP
 *          1    FN    TP
 *
 *  Flattened, implies C-contiguous, we have:
 *  0 TN
 *  1 FP
 *  2 FN
 *  3 TP
 */

namespace mmu {
namespace core {

/* Compute the comparison tolerance for a classification threshold.
 *
 * The tolerance is `atol + rtol * |threshold|`.
 */
template <typename T, isFloat<T> = true>
inline T scaled_tolerance(
    const T threshold,
    const double rtol = 1e-05,
    const double atol = 1e-8) {
    return static_cast<T>(
        atol + rtol * std::abs(static_cast<double>(threshold)));
}

/* Check if a is greater or equal to b taking into account floating point noise
 *
 * Returns true if a >= b - tolerance, i.e., a is not significantly less than b.
 *
 * Note that this function is asymmetric for the equality check as it uses
 * the scale of `b` to determine the tolerance.
 */
template <typename T1, typename T2, isFloat<T1> = true, isFloat<T2> = true>
inline bool greater_equal_tol(
    const T1 a,
    const T2 b,
    const double rtol = 1e-05,
    const double atol = 1e-8) {
    const double delta = a - b;
    const double scaled_tol = atol + rtol * std::abs(static_cast<double>(b));
    return delta >= -scaled_tol;
}

namespace details {

/* Accumulate a binary confusion matrix from two predicates. */
template <typename T1, typename T2, typename PosFn, typename PredFn>
inline void confusion_matrix_impl(
    const int64_t n_obs,
    const T1* __restrict y,
    const T2* __restrict yhat,
    int64_t* __restrict const conf_mat,
    PosFn is_actual_positive,
    PredFn is_predicted_positive) {
    int64_t n_actual_pos = 0;
    int64_t n_pred_pos = 0;
    int64_t n_true_pos = 0;

    for (int64_t i = 0; i < n_obs; ++i) {
        const int64_t a = is_actual_positive(y[i]) ? 1 : 0;
        const int64_t b = is_predicted_positive(yhat[i]) ? 1 : 0;
        n_actual_pos += a;
        n_pred_pos += b;
        n_true_pos += a & b;
    }

    conf_mat[3] = n_true_pos;                                      // TP
    conf_mat[2] = n_actual_pos - n_true_pos;                       // FN
    conf_mat[1] = n_pred_pos - n_true_pos;                         // FP
    conf_mat[0] = n_obs - n_actual_pos - n_pred_pos + n_true_pos;  // TN
}

/* Predicate: an integral label is positive when it is non-zero. */
template <typename T>
struct IntLabelPositive {
    inline bool operator()(const T v) const { return static_cast<bool>(v); }
};

/* Predicate: a floating-point label is positive when it exceeds epsilon.
 * Matches the original `y[i] > std::numeric_limits<T>::epsilon()` test.
 */
template <typename T>
struct FloatLabelPositive {
    inline bool operator()(const T v) const {
        return v > std::numeric_limits<T>::epsilon();
    }
};

/* Predicate: a score is a positive prediction when it meets the threshold,
 * allowing for floating-point noise. Tolerance computed per comparison.
 */
template <typename T>
struct ScoreAboveThreshold {
    T threshold;
    inline bool operator()(const T v) const {
        return greater_equal_tol(v, threshold);
    }
};

/* Predicate: as above but with a pre-computed tolerance, hoisted out of the
 * loop (and, for the threshold sweeps, out of the parallel region).
 */
template <typename T>
struct ScoreAboveThresholdTol {
    T threshold;
    T scaled_tol;
    inline bool operator()(const T v) const {
        return v - threshold >= -scaled_tol;
    }
};

}  // namespace details

/* Fill binary confusion matrix based on true labels y and estimated labels yhat
 *
 * --- NOTE ---
 * - this function:
 * * does not handle nullptrs
 * * expects all memory to be contiguous
 * --- NOTE ---
 *
 * --- Parameters ---
 * - n_obs : number of observations in y and yhat
 * - y : true labels
 * - yhat : estimated labels
 * - conf_mat : allocated memory for the confusion matrix
 */
inline void confusion_matrix(
    const int64_t n_obs,
    const bool* __restrict y,
    const bool* __restrict yhat,
    int64_t* __restrict const conf_mat) {
    details::confusion_matrix_impl(
        n_obs,
        y,
        yhat,
        conf_mat,
        details::IntLabelPositive<bool>{},
        details::IntLabelPositive<bool>{});
}

/* Fill binary confusion matrix from integral labels and integral predictions.
 */
template <typename T1, typename T2, isInt<T1> = true, isInt<T2> = true>
inline void confusion_matrix(
    const int64_t n_obs,
    const T1* __restrict y,
    const T2* __restrict yhat,
    int64_t* __restrict const conf_mat) {
    details::confusion_matrix_impl(
        n_obs,
        y,
        yhat,
        conf_mat,
        details::IntLabelPositive<T1>{},
        details::IntLabelPositive<T2>{});
}

/* Fill binary confusion matrix from floating-point labels and predictions. */
template <typename T1, typename T2, isFloat<T1> = true, isFloat<T2> = true>
inline void confusion_matrix(
    const int64_t n_obs,
    const T1* __restrict y,
    const T2* __restrict yhat,
    int64_t* __restrict const conf_mat) {
    details::confusion_matrix_impl(
        n_obs,
        y,
        yhat,
        conf_mat,
        details::FloatLabelPositive<T1>{},
        details::FloatLabelPositive<T2>{});
}

/* Fill binary confusion matrix from integral labels and classifier scores.
 *
 * --- Parameters ---
 * - threshold : inclusive classification threshold
 */
template <typename T1, typename T2, isInt<T1> = true, isFloat<T2> = true>
inline void confusion_matrix(
    const int64_t n_obs,
    const T1* __restrict y,
    const T2* __restrict score,
    const T2 threshold,
    int64_t* __restrict const conf_mat) {
    details::confusion_matrix_impl(
        n_obs,
        y,
        score,
        conf_mat,
        details::IntLabelPositive<T1>{},
        details::ScoreAboveThreshold<T2>{threshold});
}

/* Fill binary confusion matrix from float labels and classifier scores. */
template <typename T1, typename T2, isFloat<T1> = true, isFloat<T2> = true>
inline void confusion_matrix(
    const int64_t n_obs,
    const T1* __restrict y,
    const T2* __restrict score,
    const T2 threshold,
    int64_t* __restrict const conf_mat) {
    details::confusion_matrix_impl(
        n_obs,
        y,
        score,
        conf_mat,
        details::FloatLabelPositive<T1>{},
        details::ScoreAboveThreshold<T2>{threshold});
}

/* Fill binary confusion matrix from integral labels and classifier scores,
 * with a pre-computed tolerance.
 *
 * --- Parameters ---
 * - threshold : inclusive classification threshold
 * - scaled_tol : pre-computed tolerance, see `scaled_tolerance`
 */
template <typename T1, typename T2, isInt<T1> = true, isFloat<T2> = true>
inline void confusion_matrix(
    const int64_t n_obs,
    const T1* __restrict y,
    const T2* __restrict score,
    const T2 threshold,
    const T2 scaled_tol,
    int64_t* __restrict const conf_mat) {
    details::confusion_matrix_impl(
        n_obs,
        y,
        score,
        conf_mat,
        details::IntLabelPositive<T1>{},
        details::ScoreAboveThresholdTol<T2>{threshold, scaled_tol});
}

/* Fill binary confusion matrix from float labels and classifier scores,
 * with a pre-computed tolerance.
 */
template <typename T1, typename T2, isFloat<T1> = true, isFloat<T2> = true>
inline void confusion_matrix(
    const int64_t n_obs,
    const T1* __restrict y,
    const T2* __restrict score,
    const T2 threshold,
    const T2 scaled_tol,
    int64_t* __restrict const conf_mat) {
    details::confusion_matrix_impl(
        n_obs,
        y,
        score,
        conf_mat,
        details::FloatLabelPositive<T1>{},
        details::ScoreAboveThresholdTol<T2>{threshold, scaled_tol});
}

// =============================================================================
// Sort-based threshold sweep
// =============================================================================

namespace details {

/* One observation, ordered by score, carrying the running positive count.
 *
 * `n_pos_below` holds the number of positive labels strictly before this
 * element once `build_sorted_scores` has run, which turns every threshold
 * query into a binary search plus four subtractions.
 */
template <typename T>
struct SortedScore {
    T score;
    int64_t n_pos_below;
};

/* Sort observations by score and convert the label column into an exclusive
 * prefix sum of positives. Returns the total number of positive labels.
 *
 * Sorting (score, label) pairs together rather than sorting an index
 * permutation keeps the comparison and the payload in the same cache line;
 * an index sort would chase `score[idx]` randomly through memory on every
 * comparison.
 */
template <typename T1, typename T2, typename PosFn>
inline int64_t build_sorted_scores(
    const int64_t n_obs,
    const T1* __restrict y,
    const T2* __restrict score,
    SortedScore<T2>* __restrict sorted,
    PosFn is_actual_positive) {
    for (int64_t i = 0; i < n_obs; ++i) {
        sorted[i].score = score[i];
        sorted[i].n_pos_below = is_actual_positive(y[i]) ? 1 : 0;
    }

    std::sort(
        sorted,
        sorted + n_obs,
        [](const SortedScore<T2>& a, const SortedScore<T2>& b) {
            return a.score < b.score;
        });

    int64_t running = 0;
    for (int64_t i = 0; i < n_obs; ++i) {
        const int64_t is_pos = sorted[i].n_pos_below;
        sorted[i].n_pos_below = running;
        running += is_pos;
    }
    return running;
}

/* Fill one confusion matrix from the prepared, sorted observations. */
template <typename T2>
inline void confusion_matrix_from_sorted(
    const int64_t n_obs,
    const int64_t n_total_pos,
    const SortedScore<T2>* __restrict sorted,
    const T2 threshold,
    const T2 scaled_tol,
    int64_t* __restrict const conf_mat) {
    // An observation is predicted positive when
    //     score - threshold >= -scaled_tol
    // i.e. when score >= threshold - scaled_tol. `cutoff` is that bound, so
    // the predicted-positive set is exactly the sorted suffix [idx, n_obs).
    const T2 cutoff = threshold - scaled_tol;

    const auto it = std::lower_bound(
        sorted,
        sorted + n_obs,
        cutoff,
        [](const SortedScore<T2>& elem, const T2 value) {
            return elem.score < value;
        });
    const int64_t idx = static_cast<int64_t>(it - sorted);

    // `n_pos_below` at `idx` is the count of positives in [0, idx); at
    // idx == n_obs the whole array precedes the cut, hence `n_total_pos`.
    const int64_t pos_below
        = idx < n_obs ? sorted[idx].n_pos_below : n_total_pos;

    const int64_t n_pred_pos = n_obs - idx;
    const int64_t tp = n_total_pos - pos_below;

    conf_mat[3] = tp;                                // TP
    conf_mat[2] = n_total_pos - tp;                  // FN
    conf_mat[1] = n_pred_pos - tp;                   // FP
    conf_mat[0] = n_obs - n_pred_pos - conf_mat[2];  // TN
}

}  // namespace details

/* Confusion matrices for many thresholds, in O((n + n_t) log n). */
template <typename T1, typename T2, typename PosFn>
inline void confusion_matrix_thresholds_sorted(
    const int64_t n_obs,
    const int64_t n_thresholds,
    const T1* __restrict y,
    const T2* __restrict score,
    const T2* __restrict thresholds,
    int64_t* __restrict const conf_mat,
    PosFn is_actual_positive) {
    std::vector<details::SortedScore<T2>> sorted(static_cast<size_t>(n_obs));

    const int64_t n_total_pos = details::build_sorted_scores(
        n_obs, y, score, sorted.data(), is_actual_positive);

    const details::SortedScore<T2>* const sorted_ptr = sorted.data();

    for (int64_t i = 0; i < n_thresholds; ++i) {
        const T2 threshold = thresholds[i];
        details::confusion_matrix_from_sorted(
            n_obs,
            n_total_pos,
            sorted_ptr,
            threshold,
            scaled_tolerance(threshold),
            conf_mat + (i * 4));
    }
}

/* Decide whether the sort-based sweep is worth its O(n) scratch allocation.
 *
 * The direct path costs ~n_thresholds * n_obs comparisons spread over
 * `n_threads`; the sorted path costs ~n_obs * log2(n_obs) for a *serial*
 * sort plus n_thresholds * log2(n_obs) for the queries. Sorting therefore
 * wins once
 *
 *      n_thresholds > k * n_threads * log2(n_obs)
 *
 * `k` absorbs the ratio between the sort's per-element cost (branch
 * misprediction, swaps, 16-byte payload movement) and the direct path's
 * trivially predictable, vectorised compare-and-add. Measured on an
 * 8-thread arm64 machine at n_obs = 1e6 the empirical crossover sits near
 * n_thresholds ~= 1400, i.e. k ~= 8.8; `k = 8.0` is used so the estimate
 * errs slightly toward the direct path, which allocates nothing.
 *
 * Being wrong in that direction costs a small constant factor near the
 * crossover; being wrong the other way reintroduces the quadratic blowup,
 * so the asymmetry is deliberate.
 */
inline bool prefer_sorted_sweep(
    const int64_t n_obs,
    const int64_t n_thresholds,
    const int n_threads = 1) {
    if (n_obs < 2048 || n_thresholds < 2) {
        return false;
    }
    const double log2_n = std::log2(static_cast<double>(n_obs));
    const double threads = static_cast<double>(n_threads > 0 ? n_threads : 1);
    return static_cast<double>(n_thresholds) > 8.0 * threads * log2_n;
}

}  // namespace core
}  // namespace mmu
