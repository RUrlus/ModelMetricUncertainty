/* multn_loglike.hpp -- Main include file for multinomial log-likelihood module
 * Copyright 2022 Ralph Urlus
 *
 * This is the refactored multinomial log-likelihood module that provides
 * template-based implementations for computing uncertainty using profile
 * log-likelihoods and Wilks' theorem.
 *
 * Usage:
 *   #include <mmu/core/multn_loglike.hpp>
 *
 * This includes:
 *   - common.hpp: Common types (prof_loglike_store)
 *   - metrics.hpp: Profile classes (PrecisionRecallProfile, ROCProfile, RecallPPNProfile)
 *   - grid_bounds.hpp: Grid bounds computation
 *   - core.hpp: Template functions (multn_chi2_score, multn_error, etc.)
 *   - pr.hpp: Precision-Recall namespace aliases
 *   - roc.hpp: ROC (TPR-FPR) namespace aliases
 *   - recall_ppn.hpp: Recall-PPN namespace aliases
 *
 * You can also include individual headers:
 *   #include <mmu/core/multn_loglike/pr.hpp>
 *   #include <mmu/core/multn_loglike/roc.hpp>
 *   #include <mmu/core/multn_loglike/recall_ppn.hpp>
 */
#ifndef INCLUDE_MMU_CORE_MULTN_LOGLIKE_HPP_
#define INCLUDE_MMU_CORE_MULTN_LOGLIKE_HPP_

// Include order matters - dependencies first
#include <mmu/core/multn_loglike/common.hpp>
#include <mmu/core/multn_loglike/metrics.hpp>
#include <mmu/core/multn_loglike/grid_bounds.hpp>
#include <mmu/core/multn_loglike/core.hpp>

// Namespace aliases for specific metric pairs
#include <mmu/core/multn_loglike/pr.hpp>
#include <mmu/core/multn_loglike/roc.hpp>
#include <mmu/core/multn_loglike/recall_ppn.hpp>

#endif  // INCLUDE_MMU_CORE_MULTN_LOGLIKE_HPP_

