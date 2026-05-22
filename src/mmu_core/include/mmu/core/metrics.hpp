/* metrics.hpp -- Implementation of binary classification metrics
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_CORE_METRICS_HPP_
#define INCLUDE_MMU_CORE_METRICS_HPP_

#include <algorithm>    // for max/min
#include <cinttypes>    // for int64_t
#include <cmath>        // for sqrt
#include <limits>       // for numeric_limits
#include <stdexcept>    // for runtime_error
#include <type_traits>  // for enable_if_t

#include <mmu/core/common.hpp>

namespace mmu {
namespace core {

inline void precision_recall(
    const int64_t* __restrict const conf_mat,
    double* __restrict const metrics,
    const double fill = 0.) {
    // real true/positive observations [FN + TP]
    const int64_t iP = conf_mat[2] + conf_mat[3];
    const bool P_nonzero = iP > 0;
    const auto P = static_cast<double>(iP);
    const auto tp = static_cast<double>(conf_mat[3]);

    const int64_t itp_fp = conf_mat[3] + conf_mat[1];
    const bool tp_fp_nonzero = itp_fp > 0;

    // metrics[0]  - pos.precision aka Positive Predictive Value (PPV)
    metrics[0] = tp_fp_nonzero ? tp / static_cast<double>(itp_fp) : fill;

    // metrics[1]  - pos.recall aka True Positive Rate (TPR) aka Sensitivity
    metrics[1] = P_nonzero ? tp / P : fill;
}  // precision_recall

inline void precision_recall_ecc(
    const int64_t* __restrict const conf_mat,
    double* __restrict const metrics) {
    // real true/positive observations [FN + TP]
    const auto itp = conf_mat[3];
    const auto tp = static_cast<double>(conf_mat[3]);

    const int64_t itp_fn = conf_mat[2] + conf_mat[3];
    const int64_t itp_fp = conf_mat[1] + conf_mat[3];

    // metrics[0]  - pos.precision aka Positive Predictive Value (PPV)
    if (itp == itp_fp) {
        metrics[0] = 1.0;
    } else if (itp_fp > 0) {
        metrics[0] = tp / static_cast<double>(itp_fp);
    } else {
        // precision == 0
        metrics[0] = 0.0;
    }

    // metrics[1]  - pos.recall aka True Positive Rate (TPR) aka Sensitivity
    if (itp == itp_fn) {
        // recall == 1
        metrics[1] = 1.0;
    } else if (itp_fn > 0) {
        metrics[1] = tp / static_cast<double>(itp_fn);
    } else {
        // recall == 0.0
        metrics[1] = 0.0;
    }
}  // precision_recall_ecc

/* Sets the following values at metrics index:
 *    3 - pos.recall aka True Positive Rate aka Sensitivity
 *    6 - False Positive Rate
 */
template <class T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
inline void ROC(
    T* __restrict const conf_mat,
    double* __restrict const metrics,
    const double fill = 0.) {
    /*
     *                  pred
     *                0     1
     *  actual  0    TN    FP
     *          1    FN    TP
     *
     *  Flattened we have:
     *  0 TN
     *  1 FP
     *  2 FN
     *  3 TP
     *
     */

    // real true/positive observations [FN + TP]
    const int64_t iP = conf_mat[2] + conf_mat[3];
    const bool P_nonzero = iP > 0;
    const auto P = static_cast<double>(iP);
    // real false/negative observations [TN + FP]
    const int64_t iN = conf_mat[0] + conf_mat[1];
    const bool N_nonzero = iN > 0;
    const auto N = static_cast<double>(iN);

    const auto tn = static_cast<double>(conf_mat[0]);
    const auto tp = static_cast<double>(conf_mat[3]);

    double itm = 0.0;

    // metrics[1]  - False positive Rate (FPR)
    itm = tn / N;
    metrics[1] = N_nonzero ? (1 - itm) : 1.0;

    // metrics[0]  - pos.recall aka True Positive Rate (TPR) aka Sensitivity
    itm = tp / P;
    metrics[0] = P_nonzero ? itm : fill;

}  // ROC

/* Sets the following values at metrics index:
 *    0 - neg.precision aka Negative Predictive Value
 *    1 - pos.precision aka Positive Predictive Value
 *    2 - neg.recall aka True Negative Rate & Specificity
 *    3 - pos.recall aka True Positive Rate aka Sensitivity
 *    4 - neg.f1 score
 *    5 - pos.f1 score
 *    6 - False Positive Rate
 *    7 - False Negative Rate
 *    8 - Accuracy
 *    9 - MCC
 */
template <class T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
inline void binary_metrics(
    T* __restrict const conf_mat,
    double* __restrict const metrics,
    const double fill = 0.) {
    // total observations
    const auto K = static_cast<double>(
        conf_mat[0] + conf_mat[1] + conf_mat[2] + conf_mat[3]);

    /*
     *                  pred
     *                0     1
     *  actual  0    TN    FP
     *          1    FN    TP
     *
     *  Flattened we have:
     *  0 TN
     *  1 FP
     *  2 FN
     *  3 TP
     *
     */

    // real true/positive observations [FN + TP]
    const int64_t iP = conf_mat[2] + conf_mat[3];
    const bool P_nonzero = iP > 0;
    const auto P = static_cast<double>(iP);
    // real false/negative observations [TN + FP]
    const int64_t iN = conf_mat[0] + conf_mat[1];
    const bool N_nonzero = iN > 0;
    const auto N = static_cast<double>(iN);

    const auto tn = static_cast<double>(conf_mat[0]);
    const auto fp = static_cast<double>(conf_mat[1]);
    const auto fn = static_cast<double>(conf_mat[2]);
    const auto tp = static_cast<double>(conf_mat[3]);

    const int64_t itp_fp = conf_mat[3] + conf_mat[1];
    const bool tp_fp_nonzero = itp_fp > 0;
    const auto tp_fp = static_cast<double>(itp_fp);

    const auto tp_tn = static_cast<double>(conf_mat[3] + conf_mat[0]);

    const int64_t itn_fn = conf_mat[0] + conf_mat[2];
    const bool tn_fn_nonzero = itn_fn > 0;
    const auto tn_fn = static_cast<double>(itn_fn);
    const auto fp_fn = static_cast<double>(conf_mat[1] + conf_mat[2]);

    double itm = 0.0;
    double itm_alt = 0.0;
    // metrics 0 - neg.precision aka Negative Predictive Value (NPV)
    metrics[0] = tn_fn_nonzero ? tn / tn_fn : fill;

    // metrics[1]  - pos.precision aka Positive Predictive Value (PPV)
    metrics[1] = tp_fp_nonzero ? tp / tp_fp : fill;

    // metrics[2]  - neg.recall aka True Negative Rate (TNR) & Specificity
    // metrics[6]  - False positive Rate (FPR)
    itm = tn / N;
    metrics[2] = N_nonzero ? itm : fill;
    metrics[6] = N_nonzero ? (1 - itm) : 1.0;

    // metrics[3]  - pos.recall aka True Positive Rate (TPR) aka Sensitivity
    // metrics[7]  - False Negative Rate (FNR)
    itm = tp / P;
    metrics[3] = P_nonzero ? itm : fill;
    metrics[7] = P_nonzero ? (1 - itm) : 1.0;

    // metrics[4]  - Negative F1 score
    itm_alt = 2 * tn;
    itm = itm_alt / (itm_alt + fp_fn);
    metrics[4] = (N_nonzero || tn_fn_nonzero) ? itm : fill;

    // metrics[5]  - Positive F1 score
    itm_alt = 2 * tp;
    itm = itm_alt / (itm_alt + fp_fn);
    metrics[5] = (P_nonzero || tp_fp_nonzero) ? itm : fill;

    // metrics[8]  - Accuracy
    metrics[8] = tp_tn / K;

    // metrics[9]  - MCC
    static constexpr double limit = std::numeric_limits<double>::epsilon();
    itm = tp_fp * P * N * tn_fn;
    metrics[9] = (itm > limit) ? (tp * tn - fp * fn) / std::sqrt(itm) : 0.0;
}  // binary_metrics

}  // namespace core
}  // namespace mmu

#endif  // INCLUDE_MMU_CORE_METRICS_HPP_
