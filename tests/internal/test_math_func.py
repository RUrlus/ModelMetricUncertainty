import numpy as np
from scipy.special import erfinv
from scipy.stats import norm

from mmu.lib._mmu_core_tests import norm_ppf
from mmu.lib._mmu_core_tests import erfinv as cpp_erfinv
from mmu.lib._mmu_core_tests import binomial_rvs
from mmu.lib._mmu_core_tests import multinomial_rvs


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


def test_binomial_rvs():
    p = 0.23
    n = 10
    size = 100000

    outp = binomial_rvs(size, n, p, seed=890714, stream=0)
    probas = outp / n
    assert np.isclose(probas.mean(), p, rtol=5e-4)
    assert np.max(outp) <= 10

    p = 0.78
    n = 1000
    size = 100000

    outp = binomial_rvs(size, n, p, seed=890714, stream=0)
    probas = outp / n
    assert np.isclose(probas.mean(), p, rtol=5e-4)
    assert np.max(outp) <= 1000


def test_multinomial_rvs():
    rng = np.random.default_rng(87326134)
    p = rng.dirichlet(np.ones(4))
    n = 10
    size=400_000

    outp = multinomial_rvs(size, n, p, seed=87326134, stream=0)
    obs_probas = (outp / n).mean(0)

    assert np.isclose(obs_probas, p, rtol=1e-3).all()
    assert np.max(outp) <= n
