import numpy as np
from scipy.special import erfinv
from scipy.stats import norm

from mmu.lib._mmu_core_tests import norm_ppf
from mmu.lib._mmu_core_tests import erfinv as cpp_erfinv


def test_erfinv():
    """Test C++ implemenation of erfinv compared to scipy."""
    N = 1000
    cnt = 0
    for i in np.random.uniform(1e-12, 1 - 1e-12, N):
        cnt += np.isclose(erfinv(i), cpp_erfinv(i), atol=1e-15)
    assert cnt == N


def test_erfinv_special_cases():
    """Test special cases of erfinv."""
    assert np.isnan(cpp_erfinv(np.nan))
    assert np.isnan(cpp_erfinv(-1.01))
    assert np.isnan(cpp_erfinv(1.01))
    assert np.isnan(cpp_erfinv(1. + 1e-12))
    assert np.isnan(cpp_erfinv(-1. - 1e-12))
    assert np.isinf(cpp_erfinv(1.))
    assert np.isneginf(cpp_erfinv(-1.))
    assert np.isinf(cpp_erfinv(1. + 1e-16))
    assert np.isinf(cpp_erfinv(-1. - 1e-16))


def test_norm_ppf():
    """Test C++ implemenation PPF of Normal dist compared to scipy."""
    N = 1000
    cnt = 0
    for _ in range(N):
        p = np.random.uniform(1e-12, 1 - 1e-12)
        mu = np.random.uniform(-1000, 1000)
        sigma = np.random.uniform(1e5, 1000)
        scp_ppf = norm(mu, sigma).ppf(p)
        ppf = norm_ppf(mu, sigma, p)
        cnt += np.isclose(ppf, scp_ppf, atol=1e-15)
    assert cnt == N
