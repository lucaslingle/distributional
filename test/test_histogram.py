import numpy as np
import pytest

from distributional.histogram import Histogram


TOLS = dict(atol=1e-4, rtol=1e-4)


def unif_probs(n):
    return np.array([1 / n for _ in range(n)])


def alt_probs(n):
    assert n % 2 == 0
    return np.array([(2 / n if i % 2 == 0 else 0.) for i in range(n)])


def unif_histogram(n):
    return Histogram(0., 1., n, unif_probs(n))


def alt_histogram(n):
    return Histogram(0., 1., n, alt_probs(n))


@pytest.mark.parametrize("hist", [unif_histogram, alt_histogram])
def test_histogram_convolve_rel_test(hist):
    h = hist(6)
    hc = h._convolve(h)
    hcs = h._convolve_slow(h)
    np.testing.assert_allclose(hc.probs, hcs.probs, **TOLS)


@pytest.mark.parametrize("probs, hist", [(unif_probs, unif_histogram)])
@pytest.mark.parametrize("start_n, end_n", [(2, 3), (3, 2)])
def test_histogram_rebin_unaligned(probs, hist, start_n, end_n):
    h = hist(start_n)
    h = h.rebin(0., 1., end_n)
    np.testing.assert_allclose(probs(end_n), h.probs, **TOLS)


@pytest.mark.parametrize("probs, hist", [(unif_probs, unif_histogram)])
@pytest.mark.parametrize("start_n, end_n", [(10, 100), (100, 10)])
def test_histogram_rebin_aligned(probs, hist, start_n, end_n):
    h = hist(start_n)
    h = h.rebin(0., 1., end_n)
    np.testing.assert_allclose(probs(end_n), h.probs, **TOLS)
