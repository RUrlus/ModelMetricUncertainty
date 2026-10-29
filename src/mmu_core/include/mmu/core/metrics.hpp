/* metrics.hpp -- Implementation of binary classification metrics
 * Copyright 2026 Ralph Urlus
 */
#pragma once

#include <cmath>        // for sqrt
#include <limits>       // for numeric_limits
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

/* Precision and recall with edge-case corrections.
 *
 * --- NOTE ---
 * This variant deliberately does NOT use the `fill` contract of
 * `binary_metrics`. A metric whose numerator equals its (zero) denominator is
 * reported as 1.0, and an otherwise-undefined metric as 0.0.
 *
 * It exists for the uncertainty estimators, where the grid must be anchored at
 * a concrete point in [0, 1] regardless of what sentinel a caller would prefer
 * for reporting. The same convention is implemented by
 * `multn::details::ratio_value`. Use `precision_recall` for reporting.
 * --- NOTE ---
 */
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
    // FPR is the complement of TNR, so an undefined TNR (== fill) implies
    // FPR == 1 - fill. See the `fill` contract on `binary_metrics`.
    itm = N_nonzero ? tn / N : 0.0;
    metrics[1] = N_nonzero ? (1 - itm) : (1.0 - fill);

    // metrics[0]  - pos.recall aka True Positive Rate (TPR) aka Sensitivity
    itm = P_nonzero ? tp / P : 0.0;
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
 *
 * --- `fill` contract ---
 * A metric whose denominator is zero is undefined and is set to `fill`.
 *
 * The two rate pairs are complements -- TNR + FPR == 1 and TPR + FNR == 1 --
 * and that identity must hold for every value of `fill`. The complements are
 * therefore derived rather than filled independently:
 *
 *      TNR undefined -> TNR = fill,  FPR = 1 - fill
 *      TPR undefined -> TPR = fill,  FNR = 1 - fill
 *
 * Filling both members of a pair with `fill` would give TNR + FPR == 2 * fill,
 * i.e. 2.0 for the default fill of 1.0, which is what this contract replaces.
 *
 * MCC is the one deliberate exception: it is set to 0.0, never `fill`. Its
 * denominator vanishes whenever any row or column of the confusion matrix is
 * empty -- a common case, e.g. a model that predicts a single class -- and
 * scikit-learn's `matthews_corrcoef` likewise returns 0.0 there. Filling it
 * with the default `fill` of 1.0 would report perfect correlation for a
 * degenerate model.
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
    itm = N_nonzero ? tn / N : 0.0;
    metrics[2] = N_nonzero ? itm : fill;
    metrics[6] = N_nonzero ? (1 - itm) : (1.0 - fill);

    // metrics[3]  - pos.recall aka True Positive Rate (TPR) aka Sensitivity
    // metrics[7]  - False Negative Rate (FNR)
    itm = P_nonzero ? tp / P : 0.0;
    metrics[3] = P_nonzero ? itm : fill;
    metrics[7] = P_nonzero ? (1 - itm) : (1.0 - fill);

    // metrics[4]  - Negative F1 score
    // The denominator 2*TN + FP + FN is zero exactly when TN, FP and FN are
    // all zero, i.e. when neither N_nonzero nor tn_fn_nonzero holds.
    itm_alt = 2 * tn;
    itm = itm_alt + fp_fn;
    metrics[4] = itm > 0.0 ? itm_alt / itm : fill;

    // metrics[5]  - Positive F1 score
    // The denominator 2*TP + FP + FN is zero exactly when TP, FP and FN are
    // all zero, i.e. when neither P_nonzero nor tp_fp_nonzero holds.
    itm_alt = 2 * tp;
    itm = itm_alt + fp_fn;
    metrics[5] = itm > 0.0 ? itm_alt / itm : fill;

    // metrics[8]  - Accuracy
    metrics[8] = K > 0.0 ? tp_tn / K : fill;

    // metrics[9]  - MCC
    // Deliberately 0.0 rather than `fill`
    static constexpr double limit = std::numeric_limits<double>::epsilon();
    itm = tp_fp * P * N * tn_fn;
    const auto mcc_numerator = static_cast<double>(
        static_cast<long double>(tp) * static_cast<long double>(tn)
        - static_cast<long double>(fp) * static_cast<long double>(fn));
    metrics[9] = (itm > limit) ? mcc_numerator / std::sqrt(itm) : 0.0;
}  // binary_metrics

}  // namespace core
}  // namespace mmu
