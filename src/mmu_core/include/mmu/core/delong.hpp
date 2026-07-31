/* delong.hpp -- Implementation of DeLong AUC covariance and paired test
 * Copyright 2026 Ralph Urlus
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(MMU_HAS_OPENMP_SUPPORT)
#include <omp.h>
#endif  // MMU_HAS_OPENMP_SUPPORT

namespace mmu {
namespace core {

struct DeLongResult {
    std::vector<double> aucs;
    std::vector<double> covariance;
};

/* Per-model AUCs and structural components (Sun & Xu, 2014).
 *
 * v10 has shape (k, n_positive) and holds V10_i = (Tz_i - Tx_i) / n_negative,
 * v01 has shape (k, n_negative) and holds V01_j = 1 - (Tz_j - Ty_j) / n_positive.
 * Both are row-major with one row per model.
 */
struct DeLongComponents {
    std::int64_t n_positive;
    std::int64_t n_negative;
    std::vector<double> aucs;
    std::vector<double> v10;
    std::vector<double> v01;
};

struct DeLongTestResult {
    double auc1;
    double auc2;
    double difference;
    double variance;
    double z;
    double p_value;
};

namespace detail {

/* Fill the structural components of one model with a single sort.
 *
 * For a positive x_i: Tz_i - Tx_i equals the number of negatives with a
 * lower score plus half the number of tied negatives, hence
 *     V10_i = (#neg < x_i + 0.5 #neg == x_i) / n.
 * Likewise for a negative y_j:
 *     V01_j = 1 - (#pos < y_j + 0.5 #pos == y_j) / m.
 * This is equivalent to the three midrank computations of Sun & Xu (2014)
 * but needs one sort of the pooled scores instead of three.
 *
 * `class_index[i]` is the position of observation `i` within its class, so
 * that components of different models are aligned per observation.
 */
inline double fill_components(
    const std::int64_t* y,
    const double* scores,
    const std::int64_t n,
    const std::int64_t n_positive,
    const std::int64_t n_negative,
    const std::int64_t* class_index,
    std::vector<std::pair<double, std::int64_t>>& sorted,
    double* v10,
    double* v01) {
    for (std::int64_t i = 0; i < n; ++i) {
        sorted[i] = {scores[i], i};
    }
    std::sort(sorted.begin(), sorted.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });

    const double m_d = static_cast<double>(n_positive);
    const double n_d = static_cast<double>(n_negative);

    std::int64_t pos_below = 0;
    std::int64_t neg_below = 0;
    double v10_sum = 0.0;

    std::int64_t first = 0;
    while (first < n) {
        std::int64_t last = first + 1;
        while (last < n && sorted[last].first == sorted[first].first) {
            ++last;
        }

        std::int64_t pos_tied = 0;
        for (std::int64_t t = first; t < last; ++t) {
            pos_tied += (y[sorted[t].second] != 0);
        }
        const std::int64_t neg_tied = (last - first) - pos_tied;

        const double v10_group
            = (static_cast<double>(neg_below) + 0.5 * static_cast<double>(neg_tied)) / n_d;
        const double v01_group
            = 1.0 - (static_cast<double>(pos_below) + 0.5 * static_cast<double>(pos_tied)) / m_d;

        for (std::int64_t t = first; t < last; ++t) {
            const std::int64_t idx = sorted[t].second;
            if (y[idx] != 0) {
                v10[class_index[idx]] = v10_group;
            } else {
                v01[class_index[idx]] = v01_group;
            }
        }
        v10_sum += static_cast<double>(pos_tied) * v10_group;

        pos_below += pos_tied;
        neg_below += neg_tied;
        first = last;
    }
    return v10_sum / m_d;
}

/* Centre each of the `k` rows of length `n` (row-major) in place.
 *
 * Two-pass centring followed by plain cross products is numerically stable
 * and lets all k(k+1)/2 covariances share the centred data. Rows are first
 * shifted by their first element so that constant rows become exactly zero,
 * keeping a degenerate (zero) variance exactly zero.
 */
inline void centre_rows(double* data, const std::int64_t k, const std::int64_t n) {
    for (std::int64_t r = 0; r < k; ++r) {
        double* row = data + r * n;
        const double shift = row[0];
        double sum = 0.0;
        for (std::int64_t i = 0; i < n; ++i) {
            row[i] -= shift;
            sum += row[i];
        }
        const double mean = sum / static_cast<double>(n);
        for (std::int64_t i = 0; i < n; ++i) {
            row[i] -= mean;
        }
    }
}

/* Number of threads to use for `n_tasks` independent tasks. */
inline int n_workers(const int n_threads, const std::int64_t n_tasks) {
#if defined(MMU_HAS_OPENMP_SUPPORT)
    const std::int64_t capped = std::min<std::int64_t>(n_threads, n_tasks);
    return static_cast<int>(std::max<std::int64_t>(capped, 1));
#else
    (void)n_threads;
    (void)n_tasks;
    return 1;
#endif  // MMU_HAS_OPENMP_SUPPORT
}

inline double dot(const double* lhs, const double* rhs, const std::int64_t n) {
    double acc = 0.0;
    for (std::int64_t i = 0; i < n; ++i) {
        acc += lhs[i] * rhs[i];
    }
    return acc;
}

}  // namespace detail

/* Compute AUCs and structural components for `k` models.
 *
 * `scores[r]` points to the `n` scores of model `r`. With OpenMP support the
 * models are processed in parallel using up to `min(n_threads, k)` threads;
 * the result does not depend on the number of threads.
 */
inline DeLongComponents delong_components(
    const std::int64_t* y,
    const double* const* scores,
    const std::int64_t n,
    const std::int64_t k,
    const int n_threads = 1) {
    if (y == nullptr || scores == nullptr) {
        throw std::invalid_argument("Input pointers must not be null.");
    }
    if (k < 1) {
        throw std::invalid_argument("`k` must be at least one.");
    }
    for (std::int64_t r = 0; r < k; ++r) {
        if (scores[r] == nullptr) {
            throw std::invalid_argument("Input pointers must not be null.");
        }
    }

    std::vector<std::int64_t> class_index(static_cast<std::size_t>(n));
    std::int64_t n_positive = 0;
    std::int64_t n_negative = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        class_index[i] = (y[i] != 0) ? n_positive++ : n_negative++;
    }

    if (n_positive < 2 || n_negative < 2) {
        throw std::invalid_argument(
            "DeLong requires at least two observations in each class.");
    }

    const auto uk = static_cast<std::size_t>(k);
    DeLongComponents result{
        n_positive,
        n_negative,
        std::vector<double>(uk),
        std::vector<double>(uk * static_cast<std::size_t>(n_positive)),
        std::vector<double>(uk * static_cast<std::size_t>(n_negative))};

    const int n_sort_workers = detail::n_workers(n_threads, k);

    // Allocate per-thread sort buffers up front so no allocation can throw
    // inside the parallel region.
    std::vector<std::vector<std::pair<double, std::int64_t>>> buffers(
        static_cast<std::size_t>(n_sort_workers),
        std::vector<std::pair<double, std::int64_t>>(static_cast<std::size_t>(n)));

#if defined(MMU_HAS_OPENMP_SUPPORT)
#pragma omp parallel for num_threads(n_sort_workers) schedule(dynamic, 1) if (n_sort_workers > 1)
#endif  // MMU_HAS_OPENMP_SUPPORT
    for (std::int64_t r = 0; r < k; ++r) {
        int tid = 0;
#if defined(MMU_HAS_OPENMP_SUPPORT)
        tid = omp_get_thread_num();
#endif  // MMU_HAS_OPENMP_SUPPORT
        result.aucs[r] = detail::fill_components(
            y,
            scores[r],
            n,
            n_positive,
            n_negative,
            class_index.data(),
            buffers[static_cast<std::size_t>(tid)],
            result.v10.data() + r * n_positive,
            result.v01.data() + r * n_negative);
    }
    return result;
}

/* Overload for row-major scores with shape (k, n). */
inline DeLongComponents delong_components(
    const std::int64_t* y,
    const double* scores,
    const std::int64_t n,
    const std::int64_t k,
    const int n_threads = 1) {
    if (scores == nullptr) {
        throw std::invalid_argument("Input pointers must not be null.");
    }
    std::vector<const double*> rows(static_cast<std::size_t>(std::max<std::int64_t>(k, 0)));
    for (std::int64_t r = 0; r < k; ++r) {
        rows[r] = scores + r * n;
    }
    return delong_components(y, rows.data(), n, k, n_threads);
}

inline DeLongResult delong(
    const std::int64_t* y,
    const double* scores,
    const std::int64_t n,
    const std::int64_t k,
    const int n_threads = 1) {
    auto comps = delong_components(y, scores, n, k, n_threads);
    const std::int64_t m = comps.n_positive;
    const std::int64_t nn = comps.n_negative;

    [[maybe_unused]] const int n_row_workers = detail::n_workers(n_threads, k);
#if defined(MMU_HAS_OPENMP_SUPPORT)
#pragma omp parallel for num_threads(n_row_workers) schedule(static) if (n_row_workers > 1)
#endif  // MMU_HAS_OPENMP_SUPPORT
    for (std::int64_t r = 0; r < k; ++r) {
        detail::centre_rows(comps.v10.data() + r * m, 1, m);
        detail::centre_rows(comps.v01.data() + r * nn, 1, nn);
    }

    const double scale10 = 1.0 / (static_cast<double>(m) * static_cast<double>(m - 1));
    const double scale01 = 1.0 / (static_cast<double>(nn) * static_cast<double>(nn - 1));

    DeLongResult result{
        comps.aucs,
        std::vector<double>(static_cast<std::size_t>(k) * static_cast<std::size_t>(k))};

    // Upper-triangular (lhs <= rhs) pairs, flattened for a single parallel loop.
    const std::int64_t n_pairs = k * (k + 1) / 2;
    std::vector<std::int64_t> pair_lhs(static_cast<std::size_t>(n_pairs));
    std::vector<std::int64_t> pair_rhs(static_cast<std::size_t>(n_pairs));
    for (std::int64_t lhs = 0, idx = 0; lhs < k; ++lhs) {
        for (std::int64_t rhs = lhs; rhs < k; ++rhs, ++idx) {
            pair_lhs[idx] = lhs;
            pair_rhs[idx] = rhs;
        }
    }

    [[maybe_unused]] const int n_pair_workers = detail::n_workers(n_threads, n_pairs);
#if defined(MMU_HAS_OPENMP_SUPPORT)
#pragma omp parallel for num_threads(n_pair_workers) schedule(static) if (n_pair_workers > 1)
#endif  // MMU_HAS_OPENMP_SUPPORT
    for (std::int64_t idx = 0; idx < n_pairs; ++idx) {
        const std::int64_t lhs = pair_lhs[idx];
        const std::int64_t rhs = pair_rhs[idx];
        const double covariance
            = scale10 * detail::dot(comps.v10.data() + lhs * m, comps.v10.data() + rhs * m, m)
              + scale01
                    * detail::dot(comps.v01.data() + lhs * nn, comps.v01.data() + rhs * nn, nn);
        result.covariance[lhs * k + rhs] = covariance;
        result.covariance[rhs * k + lhs] = covariance;
    }

    return result;
}

/* Paired two-sided DeLong test for AUC1 - AUC2.
 *
 * The variance is computed from the differenced structural components,
 * Var(V10_1 - V10_2) / m + Var(V01_1 - V01_2) / n, which equals
 * S11 + S22 - 2 S12 but avoids catastrophic cancellation for strongly
 * correlated models and is non-negative by construction.
 *
 * If the variance is zero while the AUCs differ the estimate is degenerate
 * and `z` and `p_value` are NaN.
 */
inline DeLongTestResult delong_test(
    const std::int64_t* y,
    const double* scores1,
    const double* scores2,
    const std::int64_t n,
    const int n_threads = 1) {
    const double* scores[2] = {scores1, scores2};
    auto comps = delong_components(y, scores, n, 2, n_threads);
    const std::int64_t m = comps.n_positive;
    const std::int64_t nn = comps.n_negative;

    // Difference in-place into the first model's rows.
    for (std::int64_t i = 0; i < m; ++i) {
        comps.v10[i] -= comps.v10[m + i];
    }
    for (std::int64_t j = 0; j < nn; ++j) {
        comps.v01[j] -= comps.v01[nn + j];
    }
    detail::centre_rows(comps.v10.data(), 1, m);
    detail::centre_rows(comps.v01.data(), 1, nn);

    const double var10 = detail::dot(comps.v10.data(), comps.v10.data(), m)
                         / static_cast<double>(m - 1);
    const double var01 = detail::dot(comps.v01.data(), comps.v01.data(), nn)
                         / static_cast<double>(nn - 1);
    const double variance = var10 / static_cast<double>(m) + var01 / static_cast<double>(nn);

    const double difference = comps.aucs[0] - comps.aucs[1];

    double z = 0.0;
    double p_value = 1.0;
    if (variance > 0.0) {
        z = difference / std::sqrt(variance);
        p_value = std::erfc(std::abs(z) / std::sqrt(2.0));
    } else if (difference != 0.0) {
        z = std::numeric_limits<double>::quiet_NaN();
        p_value = std::numeric_limits<double>::quiet_NaN();
    }

    return {comps.aucs[0], comps.aucs[1], difference, variance, z, p_value};
}

}  // namespace core
}  // namespace mmu

