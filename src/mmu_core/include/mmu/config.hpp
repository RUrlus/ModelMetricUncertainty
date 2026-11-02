#pragma once

#include <cstdint>

namespace mmu {

constexpr double MULT_DEFAULT_EPSILON = 1e-12;
constexpr double MULT_DEFAULT_N_SIGMAS = 6.0;

/* Minimum work before entering an OpenMP region.*/

// Per-point profile log-likelihood evaluations (`multn_chi2_scores_mt`).
constexpr int64_t MMU_OMP_MIN_POINTS = 1024;

// Grid elements in a reduction (`multn_grid_curve_error_mt`).
constexpr int64_t MMU_OMP_MIN_GRID_ELEM = 4096;

// Independent confusion matrices (`confusion_matrix_runs`, threshold sweeps).
constexpr int64_t MMU_OMP_MIN_TASKS = 2;

}  // namespace mmu
