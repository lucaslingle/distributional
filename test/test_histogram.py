import numpy as np
import pytest

from distributional.histogram import Histogram

@pytest.fixture
def unif_probs():
    def func(n):
        return np.array([1 / n for _ in range(n)])
    return func

@pytest.fixture
def unif_histogram(unif_probs):
    def func(n):
        return Histogram(0., 1., n, unif_probs(n))
    return func

def test_histogram_convolve_rel_test(unif_histogram):
    hist = unif_histogram(6)
    hist_conv = hist._convolve(hist)
    hist_conv_slow = hist._convolve_slow(hist)
    np.testing.assert_allclose(hist_conv.probs, hist_conv_slow.probs)

def test_histogram_rebin_unaligned_low2hi(unif_probs, unif_histogram):
    hist = unif_histogram(2)
    hist = hist.rebin(0., 1., 3)
    np.testing.assert_allclose(unif_probs(3), hist.probs)

def test_histogram_rebin_unaligned_hi2low(unif_probs, unif_histogram):
    hist = unif_histogram(3)
    hist = hist.rebin(0., 1., 2)
    np.testing.assert_allclose(unif_probs(2), hist.probs)

def test_histogram_rebin_aligned_low2hi(unif_probs, unif_histogram):
    hist = unif_histogram(10)
    hist = hist.rebin(0., 1., 100)
    np.testing.assert_allclose(unif_probs(100), hist.probs)

def test_histogram_rebin_aligned_hi2low(unif_probs, unif_histogram):
    hist = unif_histogram(100)
    hist = hist.rebin(0., 1., 10)
    np.testing.assert_allclose(unif_probs(10), hist.probs)
