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
        return static_cast<double>(conf_mat[1] + conf_mat[2] + conf_mat[3]);
    }

    static inline void
    metric_values(const int64_t* __restrict conf_mat, double& y, double& x) {
        const double fp = static_cast<double>(conf_mat[1]);
        const double fn = static_cast<double>(conf_mat[2]);
        const double tp = static_cast<double>(conf_mat[3]);

        y = details::safe_ratio(tp, tp + fp);  // precision
        x = details::safe_ratio(tp, tp + fn);  // recall
    }

    static inline void metric_sigmas(
        const int64_t* __restrict conf_mat,
        double& y_sigma,
        double& x_sigma) {
        double y, x;
        metric_values(conf_mat, y, x);

        const double pred_pos = static_cast<double>(conf_mat[1] + conf_mat[3]);
        const double pos = static_cast<double>(conf_mat[2] + conf_mat[3]);

        y_sigma = details::safe_binom_sigma(y, pred_pos);
        x_sigma = details::safe_binom_sigma(x, pos);
    }

    template <bool Guarded>
    static inline bool constrained_fit(
        const double y,  // precision
        const double x,  // recall
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
        (void)epsilon;
        max_y_clip = conf_mat[1] == 0 ? 0.0 : epsilon;
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
        return static_cast<double>(conf_mat[2] + conf_mat[3]);
    }

    static inline void
    metric_values(const int64_t* __restrict conf_mat, double& y, double& x) {
        const double tn = static_cast<double>(conf_mat[0]);
        const double fp = static_cast<double>(conf_mat[1]);
        const double fn = static_cast<double>(conf_mat[2]);
        const double tp = static_cast<double>(conf_mat[3]);

        y = details::safe_ratio(tp, tp + fn);  // TPR
        x = details::safe_ratio(fp, fp + tn);  // FPR
    }

    static inline void metric_sigmas(
        const int64_t* __restrict conf_mat,
        double& y_sigma,
        double& x_sigma) {
        double y, x;
        metric_values(conf_mat, y, x);

        const double pos = static_cast<double>(conf_mat[2] + conf_mat[3]);
        const double neg = static_cast<double>(conf_mat[0] + conf_mat[1]);

        y_sigma = details::safe_binom_sigma(y, pos);
        x_sigma = details::safe_binom_sigma(x, neg);
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
        (void)epsilon;
        max_y_clip = conf_mat[2] == 0 ? 0.0 : epsilon;
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
        return static_cast<double>(conf_mat[2] + conf_mat[3]);  // FN + TP
    }

    static inline void
    metric_values(const int64_t* __restrict conf_mat, double& y, double& x) {
        const double tn = static_cast<double>(conf_mat[0]);
        const double fn = static_cast<double>(conf_mat[2]);
        const double tp = static_cast<double>(conf_mat[3]);
        const double n = static_cast<double>(
            conf_mat[0] + conf_mat[1] + conf_mat[2] + conf_mat[3]);

        y = (tn + fn) / n;                     // PPN
        x = details::safe_ratio(tp, tp + fn);  // Recall
    }

    static inline void metric_sigmas(
        const int64_t* __restrict conf_mat,
        double& y_sigma,
        double& x_sigma) {
        double y, x;
        metric_values(conf_mat, y, x);

        const double n = static_cast<double>(
            conf_mat[0] + conf_mat[1] + conf_mat[2] + conf_mat[3]);
        const double pos = static_cast<double>(conf_mat[2] + conf_mat[3]);

        y_sigma = details::safe_binom_sigma(y, n);
        x_sigma = details::safe_binom_sigma(x, pos);
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

        const double k = (1.0 - x) / x;
        const double q = 1.0 - y;
        const double m = fn + tp;

        // (k n) t^2 - L t + c = 0, t = p_tp
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
