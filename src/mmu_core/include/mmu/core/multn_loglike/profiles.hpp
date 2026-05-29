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

inline double safe_ratio(const double num, const double den) {
    return den > 0.0 ? num / den : 0.0;
}

inline double safe_binom_sigma(const double p, const double n) {
    return n > 0.0 ? std::sqrt((p * (1.0 - p)) / n) : 0.0;
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
     * Exact old-style BVN sigma logic for Precision-Recall.
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
        const int64_t itp = conf_mat[3];
        const double tp = static_cast<double>(itp);

        const int64_t itp_fn = conf_mat[2] + conf_mat[3];
        const double tp_fn = static_cast<double>(itp_fn);

        const int64_t itp_fp = conf_mat[1] + conf_mat[3];
        const double tp_fp = static_cast<double>(itp_fp);

        // ---------------------------------------------------------------------
        // Precision = TP / (TP + FP)
        // ---------------------------------------------------------------------
        double prec;
        double prec_sigma;

        if (itp == itp_fp) {
            // precision == 1
            prec = 1.0;
            const double prec_for_sigma
                = static_cast<double>(itp_fp - 1) / tp_fp;
            prec_sigma
                = std::sqrt((prec_for_sigma * (1.0 - prec_for_sigma)) / tp_fp);
        } else if (itp_fp > 0) {
            prec = tp / tp_fp;
            prec_sigma = std::sqrt(
                static_cast<double>(conf_mat[3] * conf_mat[1])
                / std::pow(tp_fp, 3.0));
        } else {
            // precision == 0
            prec = 0.0;
            const double prec_for_sigma = 1.0 / tp_fp;
            prec_sigma
                = std::sqrt((prec_for_sigma * (1.0 - prec_for_sigma)) / tp_fp);
        }

        // ---------------------------------------------------------------------
        // Recall = TP / (TP + FN)
        // ---------------------------------------------------------------------
        double rec;
        double rec_sigma;

        if (itp == itp_fn) {
            // recall == 1
            rec = 1.0;
            const double rec_for_sigma
                = static_cast<double>(itp_fn - 1) / tp_fn;
            rec_sigma
                = std::sqrt((rec_for_sigma * (1.0 - rec_for_sigma)) / tp_fn);
        } else if (itp_fn > 0) {
            rec = tp / tp_fn;
            rec_sigma = std::sqrt(
                static_cast<double>(conf_mat[3] * conf_mat[2])
                / std::pow(tp_fn, 3.0));
        } else {
            // recall == 0
            rec = 0.0;
            const double rec_for_sigma = 1.0 / tp_fn;
            rec_sigma
                = std::sqrt((rec_for_sigma * (1.0 - rec_for_sigma)) / tp_fn);
        }

        out[0] = prec;
        out[1] = prec_sigma;
        out[2] = rec;
        out[3] = rec_sigma;
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
     * Exact old-style BVN sigma logic for ROC.
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
        const double sum_y = term1_y + term2_y;
        const double y = term1_y / sum_y;

        if (term1_y == 0.0) {
            term1_y = 1.0;
        }
        if (term2_y == 0.0) {
            term2_y = 1.0;
        }

        const double y_sigma
            = std::sqrt((term1_y * term2_y) / std::pow(term1_y + term2_y, 3.0));

        // ---------------------------------------------------------------------
        // X = FPR = FP / (FP + TN)
        // ---------------------------------------------------------------------
        double term1_x = static_cast<double>(conf_mat[1]);  // FP
        double term2_x = static_cast<double>(conf_mat[0]);  // TN
        const double sum_x = term1_x + term2_x;
        const double x = term1_x / sum_x;

        if (term1_x == 0.0) {
            term1_x = 1.0;
        }
        if (term2_x == 0.0) {
            term2_x = 1.0;
        }

        const double x_sigma
            = std::sqrt((term1_x * term2_x) / std::pow(term1_x + term2_x, 3.0));

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
     * Output layout:
     *   out[0] = Y = PPN
     *   out[1] = sigma(Y)
     *   out[2] = X = Recall
     *   out[3] = sigma(X)
     */
    static inline void bvn_sigma(
        const int64_t* __restrict conf_mat,
        double* __restrict out) {
        // Y = PPN = (TN + FN) / N
        // Write as:
        //   Y = A / (A + B)
        // with
        //   A = TN + FN
        //   B = FP + TP
        //
        // Interior variance:
        //   Var(Y) = A * B / (A + B)^3
        const int64_t iA = conf_mat[0] + conf_mat[2];  // TN + FN
        const int64_t iB = conf_mat[1] + conf_mat[3];  // FP + TP

        const double A = static_cast<double>(iA);
        const double B = static_cast<double>(iB);
        const double N = A + B;

        const double ppn = A / N;

        double ppn_sigma;
        if (iB == 0) {
            // PPN == 1
            const double ppn_for_sigma = (N - 1.0) / N;
            ppn_sigma = std::sqrt((ppn_for_sigma * (1.0 - ppn_for_sigma)) / N);
        } else if (iA == 0) {
            // PPN == 0
            const double ppn_for_sigma = 1.0 / N;
            ppn_sigma = std::sqrt((ppn_for_sigma * (1.0 - ppn_for_sigma)) / N);
        } else {
            // Interior exact ratio variance
            ppn_sigma = std::sqrt((A * B) / std::pow(N, 3.0));
        }

        // ---------------------------------------------------------------------
        // X = Recall = TP / (TP + FN)
        // ---------------------------------------------------------------------
        const int64_t itp = conf_mat[3];
        const double tp = static_cast<double>(itp);

        const int64_t itp_fn = conf_mat[2] + conf_mat[3];
        const double tp_fn = static_cast<double>(itp_fn);

        double recall;
        double recall_sigma;

        if (itp == itp_fn) {
            // Recall == 1
            recall = 1.0;
            const double recall_for_sigma
                = static_cast<double>(itp_fn - 1) / tp_fn;
            recall_sigma = std::sqrt(
                (recall_for_sigma * (1.0 - recall_for_sigma)) / tp_fn);
        } else if (itp_fn > 0) {
            recall = tp / tp_fn;
            recall_sigma = std::sqrt(
                static_cast<double>(conf_mat[3] * conf_mat[2])
                / std::pow(tp_fn, 3.0));
        } else {
            // Recall == 0
            recall = 0.0;
            const double recall_for_sigma = 1.0 / tp_fn;
            recall_sigma = std::sqrt(
                (recall_for_sigma * (1.0 - recall_for_sigma)) / tp_fn);
        }

        out[0] = ppn;
        out[1] = ppn_sigma;
        out[2] = recall;
        out[3] = recall_sigma;
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

        // smaller root, stable form
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
