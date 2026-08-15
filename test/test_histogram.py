import numpy as np
import pytest

from distributional.histogram import Histogram


TOLS = dict(atol=1e-4, rtol=1e-4)


def unif_probs(n, **kwargs):
    return np.array([1 / n for _ in range(n)])


def alt_probs(n, consec=1, **kwargs):
    assert consec >= 0
    if consec > 0:
        mass = np.array([(1. if (i // consec) % 2 == 0 else 0.) for i in range(n)])
        return Histogram._clean(mass)
    else:
        return unif_probs(n)


def unif_histogram(n, **kwargs):
    return Histogram(0., 1., n, unif_probs(n))


def alt_histogram(n, consec=1, **kwargs):
    return Histogram(0., 1., n, alt_probs(n, consec))


@pytest.mark.parametrize("hist", [unif_histogram, alt_histogram])
def test_histogram_convolve_rel_test(hist):
    h = hist(6)
    hc = h._convolve(h)
    hcs = h._convolve_slow(h)
    np.testing.assert_allclose(hc.probs, hcs.probs, **TOLS)


@pytest.mark.parametrize("probs, hist", [(unif_probs, unif_histogram)])
@pytest.mark.parametrize("start_n, end_n", [(2, 3), (3, 2)])
def test_histogram_rebin_unaligned_unif(probs, hist, start_n, end_n):
    h = hist(start_n)
    h = h.rebin(0., 1., end_n)
    np.testing.assert_allclose(probs(end_n), h.probs, **TOLS)


def test_histogram_rebin_unaligned_alt():
    h = alt_histogram(2)
    h = h.rebin(0., 1., 3)
    np.testing.assert_allclose(np.array([2/3, 1/3, 0/3]), h.probs, **TOLS)

    h = alt_histogram(3)
    h = h.rebin(0., 1., 2)
    np.testing.assert_allclose(np.array([1/2, 1/2]), h.probs, **TOLS)


@pytest.mark.parametrize("probs, hist", [(unif_probs, unif_histogram)])
@pytest.mark.parametrize("start_n, end_n", [(10, 100), (100, 10)])
def test_histogram_rebin_aligned_unif(probs, hist, start_n, end_n):
    h = hist(start_n)
    h = h.rebin(0., 1., end_n)
    np.testing.assert_allclose(probs(end_n), h.probs, **TOLS)


@pytest.mark.parametrize("probs, hist", [(alt_probs, alt_histogram)])
@pytest.mark.parametrize("start_n, end_n", [(10, 20), (10, 100), (20, 10), (100, 10)])
def test_histogram_rebin_aligned_alt(probs, hist, start_n, end_n):
    h = hist(start_n)
    h = h.rebin(0., 1., end_n)
    np.testing.assert_allclose(probs(end_n, end_n // start_n), h.probs, **TOLS)
