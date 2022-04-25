/* metrics.hpp -- Implementation of binary classification metrics
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_CORE_METRICS_HPP_
#define INCLUDE_MMU_CORE_METRICS_HPP_

#include <cmath>      // for sqrt
#include <limits>     // for numeric_limits
#include <cinttypes>  // for int64_t
#include <algorithm>  // for max/min
#include <stdexcept>  // for runtime_error
#include <type_traits>  // for enable_if_t

#include <mmu/core/common.hpp>

namespace mmu {
namespace core {

inline void precision_recall(
    int64_t* __restrict const conf_mat,
    double* __restrict const metrics,
    const double fill = 0.
) {
    // real true/positive observations [FN + TP]
    const int64_t iP = conf_mat[2] + conf_mat[3];
    const bool P_nonzero = iP > 0;
    const auto P = static_cast<double>(iP);
    const auto TP = static_cast<double>(conf_mat[3]);

    const int64_t iTP_FP = conf_mat[3] + conf_mat[1];
    const bool TP_FP_nonzero = iTP_FP > 0;

    // metrics[0]  - pos.precision aka Positive Predictive Value (PPV)
    metrics[0] = TP_FP_nonzero ? TP / static_cast<double>(iTP_FP) : fill;

    // metrics[1]  - pos.recall aka True Positive Rate (TPR) aka Sensitivity
    metrics[1] = P_nonzero ? TP / P : fill;
}  // binary_metrics

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
template<class T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
inline void binary_metrics(
    T* __restrict const conf_mat,
    double* __restrict const metrics,
    const double fill = 0.
) {
    // total observations
    const auto K = static_cast<double>(
        conf_mat[0]
        + conf_mat[1]
        + conf_mat[2]
        + conf_mat[3]
    );

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

    const auto TN = static_cast<double>(conf_mat[0]);
    const auto FP = static_cast<double>(conf_mat[1]);
    const auto FN = static_cast<double>(conf_mat[2]);
    const auto TP = static_cast<double>(conf_mat[3]);

    const int64_t iTP_FP = conf_mat[3] + conf_mat[1];
    const bool TP_FP_nonzero = iTP_FP > 0;
    const auto TP_FP = static_cast<double>(iTP_FP);

    const auto TP_TN = static_cast<double>(conf_mat[3] + conf_mat[0]);

    const int64_t iTN_FN = conf_mat[0] + conf_mat[2];
    const bool TN_FN_nonzero = iTN_FN > 0;
    const auto TN_FN = static_cast<double>(iTN_FN);
    const auto FP_FN = static_cast<double>(conf_mat[1] + conf_mat[2]);

    double itm = 0.0;
    double itm_alt = 0.0;
    // metrics 0 - neg.precision aka Negative Predictive Value (NPV)
    metrics[0] = TN_FN_nonzero ? TN / TN_FN : fill;

    // metrics[1]  - pos.precision aka Positive Predictive Value (PPV)
    metrics[1] = TP_FP_nonzero ? TP / TP_FP : fill;

    // metrics[2]  - neg.recall aka True Negative Rate (TNR) & Specificity
    // metrics[6]  - False positive Rate (FPR)
    itm = TN / N;
    metrics[2] = N_nonzero ? itm : fill;
    metrics[6] = N_nonzero ? (1 - itm) : 1.0;

    // metrics[3]  - pos.recall aka True Positive Rate (TPR) aka Sensitivity
    // metrics[7]  - False Negative Rate (FNR)
    itm = TP / P;
    metrics[3] = P_nonzero ? itm : fill;
    metrics[7] = P_nonzero ? (1 - itm) : 1.0;

    // metrics[4]  - Negative F1 score
    itm_alt = 2 * TN;
    itm = itm_alt / (itm_alt + FP_FN);
    metrics[4] = (N_nonzero || TN_FN_nonzero) ? itm : fill;

    // metrics[5]  - Positive F1 score
    itm_alt = 2 * TP;
    itm = itm_alt / (itm_alt + FP_FN);
    metrics[5] = (P_nonzero || TP_FP_nonzero) ? itm : fill;

    // metrics[8]  - Accuracy
    metrics[8] = TP_TN / K;

    // metrics[9]  - MCC
    static constexpr double limit = std::numeric_limits<double>::epsilon();
    itm = TP_FP * P * N * TN_FN;
    metrics[9] = (itm > limit) ? (TP * TN - FP * FN) / std::sqrt(itm) : 0.0;
}  // binary_metrics


}  // namespace core
}  // namespace mmu

#endif  // INCLUDE_MMU_CORE_METRICS_HPP_
