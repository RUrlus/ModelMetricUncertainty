#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

#include <mmu/core/common.hpp>
#include <mmu/core/multn_loglike/common.hpp>

namespace mmu {
namespace core {
namespace multn {

namespace details {

constexpr double k_prob_tol_mult = 128.0;
constexpr double k_disc_tol = 64.0 * std::numeric_limits<double>::epsilon();
constexpr double k_prob_tol
    = -1 * k_prob_tol_mult * std::numeric_limits<double>::epsilon();

template <bool Guarded>
inline bool neg_clip(
    const double p_tn,
    const double p_fp,
    const double p_fn,
    const double p_tp,
    double* __restrict probas) {
    if constexpr (Guarded) {
        if (p_tn < k_prob_tol || p_fp < k_prob_tol || p_fn < k_prob_tol
            || p_tp < k_prob_tol) {
            return false;
        }
    }
    probas[0] = p_tn < 0.0 ? 0.0 : p_tn;
    probas[1] = p_fp < 0.0 ? 0.0 : p_fp;
    probas[2] = p_fn < 0.0 ? 0.0 : p_fn;
    probas[3] = p_tp < 0.0 ? 0.0 : p_tp;
    return true;
}

/* Sigma of a ratio carrying no information, i.e. 0-of-0 or 1-of-1.
 *
 * Obtained by add-one smoothing: treating the observation as 1-of-2 gives
 * A = B = 1, n = 2 and therefore sqrt(A * B / n^3) = sqrt(1/8). Combined with
 * the default `n_sigmas` this widens the grid to the full [0, 1] support,
 * which is the honest representation of "we have no data".
 */
constexpr double k_no_info_sigma = 0.3535533905932737622;  // sqrt(1/8)

/* Point estimate of the ratio p = A / (A + B).
 *
 * When A + B == 0 the ratio is undefined; we return 1.0 to match the
 * convention used by `core::precision_recall_ecc`, where a metric whose
 * numerator equals its (zero) denominator is reported as 1.
 */
inline double ratio_value(const double A, const double B) {
    const double n = A + B;
    return n > 0.0 ? A / n : 1.0;
}

/* Marginal sigma of the ratio p = A / (A + B) under a binomial model.
 *
 * The plug-in variance p * (1 - p) / n collapses to exactly zero at the
 * p == 0 and p == 1 boundaries, which would yield a zero-width interval and,
 * downstream, an empty or inverted grid range. At those boundaries we
 * therefore substitute the nearest interior proportion -- 1/n for p == 0 and
 * (n - 1)/n for p == 1 -- so the interval keeps a sensible width.
 *
 * n <= 1 carries no information and returns `k_no_info_sigma`.
 *
 * The return value is always strictly positive and finite, for every possible
 * confusion matrix.
 */
inline double ratio_sigma(const double A, const double B) {
    const double n = A + B;
    if (n <= 1.0) {
        return k_no_info_sigma;
    }
    if (A <= 0.0) {
        // p == 0 boundary
        const double p = 1.0 / n;
        return std::sqrt((p * (1.0 - p)) / n);
    }
    if (B <= 0.0) {
        // p == 1 boundary
        const double p = (n - 1.0) / n;
        return std::sqrt((p * (1.0 - p)) / n);
    }
    // interior
    return std::sqrt((A * B) / (n * n * n));
}

}  // namespace details

// =============================================================================
// Precision-Recall
// =============================================================================

struct PrecisionRecallProfile {
    static constexpr const char* y_name = "Precision";
    static constexpr const char* x_name = "Recall";

    static inline double compute_metric_n(const int64_t* __restrict conf_mat) {
        // n3 = FP + FN + TP
        return static_cast<double>(conf_mat[1] + conf_mat[2] + conf_mat[3]);
    }

    /*
     * BVN sigma logic for Precision-Recall.
     *
     * Both metrics are ratios of the form A / (A + B), so the boundary and
     * no-data handling is delegated to `details::ratio_value` /
     * `details::ratio_sigma`.
     *
     *   Precision = TP / (TP + FP)
     *   Recall    = TP / (TP + FN)
     *
     * Output layout:
     *   out[0] = Precision
     *   out[1] = sigma(Precision)
     *   out[2] = Recall
     *   out[3] = sigma(Recall)
     */
    static inline void bvn_sigma(
        const int64_t* __restrict conf_mat,
        double* __restrict out) {
        const auto tp = static_cast<double>(conf_mat[3]);
        const auto fp = static_cast<double>(conf_mat[1]);
        const auto fn = static_cast<double>(conf_mat[2]);

        // Precision = TP / (TP + FP)
        out[0] = details::ratio_value(tp, fp);
        out[1] = details::ratio_sigma(tp, fp);

        // Recall = TP / (TP + FN)
        out[2] = details::ratio_value(tp, fn);
        out[3] = details::ratio_sigma(tp, fn);
    }

    template <bool Guarded>
    static inline bool constrained_fit(
        const double y,  // Precision
        const double x,  // Recall
        const prof_loglike_store& store,
        double* __restrict probas) {
        const double rec_ratio = (1.0 - x) / x;
        const double prec_ratio = (1.0 - y) / y;
        const double inv_alpha = 1.0 / (1.0 + prec_ratio + rec_ratio);

        const double p_tp = (store.metric_n / store.n) * inv_alpha;
        const double p_fn = rec_ratio * p_tp;
        const double p_fp = prec_ratio * p_tp;
        const double p_tn = 1.0 - p_fp - p_fn - p_tp;

        return details::neg_clip<Guarded>(p_tn, p_fp, p_fn, p_tp, probas);
    }

    static inline void get_max_clips(
        const int64_t* __restrict conf_mat,
        const double epsilon,
        double& max_y_clip,
        double& max_x_clip) {
        // Precision can be exactly 1 iff FP == 0
        max_y_clip = conf_mat[1] == 0 ? 0.0 : epsilon;
        // Recall can be exactly 1 iff FN == 0
        max_x_clip = conf_mat[2] == 0 ? 0.0 : epsilon;
    }
};

// =============================================================================
// ROC
// =============================================================================

struct ROCProfile {
    static constexpr const char* y_name = "TPR";
    static constexpr const char* x_name = "FPR";

    static inline double compute_metric_n(const int64_t* __restrict conf_mat) {
        // n2 = FN + TP
        return static_cast<double>(conf_mat[2] + conf_mat[3]);
    }

    /*
     * BVN sigma logic for ROC.
     *
     *   TPR = TP / (TP + FN)
     *   FPR = FP / (FP + TN)
     *
     * NOTE: this profile smooths the p == 0 / p == 1 boundaries by
     * substituting a zero term with one, rather than by shifting the
     * proportion as `details::ratio_sigma` does. The two conventions agree
     * for the fully undefined 0-of-0 case (both yield sqrt(1/8)) but differ
     * slightly at the single-sided boundaries.
     *
     * Output layout:
     *   out[0] = TPR
     *   out[1] = sigma(TPR)
     *   out[2] = FPR
     *   out[3] = sigma(FPR)
     */
    static inline void bvn_sigma(
        const int64_t* __restrict conf_mat,
        double* __restrict out) {
        // ---------------------------------------------------------------------
        // Y = TPR = TP / (TP + FN)
        // ---------------------------------------------------------------------
        double term1_y = static_cast<double>(conf_mat[3]);  // TP
        double term2_y = static_cast<double>(conf_mat[2]);  // FN
        const double y = details::ratio_value(term1_y, term2_y);

        if (term1_y == 0.0) {
            term1_y = 1.0;
        }
        if (term2_y == 0.0) {
            term2_y = 1.0;
        }

        const double sum_y = term1_y + term2_y;
        const double y_sigma
            = std::sqrt((term1_y * term2_y) / (sum_y * sum_y * sum_y));

        // ---------------------------------------------------------------------
        // X = FPR = FP / (FP + TN)
        // ---------------------------------------------------------------------
        double term1_x = static_cast<double>(conf_mat[1]);  // FP
        double term2_x = static_cast<double>(conf_mat[0]);  // TN
        const double x = details::ratio_value(term1_x, term2_x);

        if (term1_x == 0.0) {
            term1_x = 1.0;
        }
        if (term2_x == 0.0) {
            term2_x = 1.0;
        }

        const double sum_x = term1_x + term2_x;
        const double x_sigma
            = std::sqrt((term1_x * term2_x) / (sum_x * sum_x * sum_x));

        out[0] = y;
        out[1] = y_sigma;
        out[2] = x;
        out[3] = x_sigma;
    }

    template <bool Guarded>
    static inline bool constrained_fit(
        const double y,  // TPR
        const double x,  // FPR
        const prof_loglike_store& store,
        double* __restrict probas) {
        const double pos = store.metric_n / store.n;
        const double neg = 1.0 - pos;

        const double p_tp = y * pos;
        const double p_fn = (1.0 - y) * pos;
        const double p_fp = x * neg;
        const double p_tn = (1.0 - x) * neg;

        return details::neg_clip<Guarded>(p_tn, p_fp, p_fn, p_tp, probas);
    }

    static inline void get_max_clips(
        const int64_t* __restrict conf_mat,
        const double epsilon,
        double& max_y_clip,
        double& max_x_clip) {
        // TPR can be exactly 1 iff FN == 0
        max_y_clip = conf_mat[2] == 0 ? 0.0 : epsilon;
        // FPR can be exactly 1 iff TN == 0
        max_x_clip = conf_mat[0] == 0 ? 0.0 : epsilon;
    }
};

// =============================================================================
// PPN-Recall
// =============================================================================
struct PPNRecallProfile {
    static constexpr const char* y_name = "PPN";
    static constexpr const char* x_name = "Recall";

    static inline double compute_metric_n(const int64_t* __restrict conf_mat) {
        return static_cast<double>(conf_mat[2] + conf_mat[3]);
    }

    /*
     * Y = PPN    = (TN + FN) / N
     * X = Recall = TP / (TP + FN)
     *
     * Both are ratios of the form A / (A + B):
     *   PPN    with A = TN + FN, B = FP + TP   (so A + B == N)
     *   Recall with A = TP,      B = FN
     *
     * Output layout:
     *   out[0] = Y = PPN
     *   out[1] = sigma(Y)
     *   out[2] = X = Recall
     *   out[3] = sigma(X)
     */
    static inline void bvn_sigma(
        const int64_t* __restrict conf_mat,
        double* __restrict out) {
        const auto A = static_cast<double>(conf_mat[0] + conf_mat[2]);  // TN+FN
        const auto B = static_cast<double>(conf_mat[1] + conf_mat[3]);  // FP+TP

        // Y = PPN = (TN + FN) / N
        out[0] = details::ratio_value(A, B);
        out[1] = details::ratio_sigma(A, B);

        const auto tp = static_cast<double>(conf_mat[3]);
        const auto fn = static_cast<double>(conf_mat[2]);

        // X = Recall = TP / (TP + FN)
        out[2] = details::ratio_value(tp, fn);
        out[3] = details::ratio_sigma(tp, fn);
    }

    template <bool Guarded>
    static inline bool constrained_fit(
        const double y,  // PPN
        const double x,  // Recall
        const prof_loglike_store& store,
        double* __restrict probas) {
        const double tn = store.x_tn;
        const double fp = store.x_fp;
        const double fn = store.x_fn;
        const double tp = store.x_tp;
        const double n = store.n;

        // Constraints:
        //   p_fn = k * p_tp,  k = (1 - x) / x
        //   p_tn = y - p_fn
        //   p_fp = (1 - y) - p_tp
        //
        // Let q = 1 - y and t = p_tp.
        // The score equation reduces to:
        //   (k n) t^2 - L t + c = 0
        //
        // with:
        //   L = m (y + k q) + fp y + tn k q
        //   c = m y q
        //   m = fn + tp
        //
        // The interior maximiser is the smaller root, evaluated stably as:
        //   t = 2 c / (L + sqrt(D))
        // where D = L^2 - 4 a c and a = k n.

        const double k = (1.0 - x) / x;
        const double q = 1.0 - y;
        const double m = fn + tp;

        const double a = k * n;
        const double L = m * (y + k * q) + fp * y + tn * k * q;
        const double c = m * y * q;

        double D = std::fma(-4.0 * a, c, L * L);

        if constexpr (Guarded) {
            if (D < 0.0) {
                const double D_tol
                    = details::k_disc_tol * (L * L + std::abs(4.0 * a * c));
                if (D > -D_tol) {
                    D = 0.0;
                } else {
                    return false;
                }
            }
        } else {
            D = D < 0.0 ? 0.0 : D;
        }

        const double sqrt_D = std::sqrt(D);
        const double denom = L + sqrt_D;
        if (denom <= 0.0) {
            return false;
        }

        const double p_tp = (2.0 * c) / denom;
        const double p_fn = k * p_tp;
        const double p_tn = y - p_fn;
        const double p_fp = q - p_tp;

        return details::neg_clip<Guarded>(p_tn, p_fp, p_fn, p_tp, probas);
    }

    static inline void get_max_clips(
        const int64_t* __restrict conf_mat,
        const double epsilon,
        double& max_y_clip,
        double& max_x_clip) {
        (void)epsilon;
        max_y_clip = (conf_mat[1] + conf_mat[3]) == 0 ? 0.0 : epsilon;
        max_x_clip = conf_mat[2] == 0 ? 0.0 : epsilon;
    }
};

}  // namespace multn
}  // namespace core
}  // namespace mmu
