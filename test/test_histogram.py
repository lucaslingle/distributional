import numpy as np
import pytest

from distributional.histogram import Histogram

TOLS = dict(atol=1e-4, rtol=1e-4)


def unif_probs(n, **kwargs):
    return np.array([1 / n for _ in range(n)])


def alt_probs(n, consec=1, **kwargs):
    assert consec >= 0
    if consec > 0:
        mass = np.array([(1.0 if (i // consec) % 2 == 0 else 0.0) for i in range(n)])
        return Histogram.renormalize(mass)
    else:
        return unif_probs(n)


def wobbly_probs(n, consec=1, **kwargs):
    assert consec >= 0
    if consec > 0:
        mass = np.array([(1.0 if (i // consec) % 2 == 0 else 0.5) for i in range(n)])
        return Histogram.renormalize(mass)
    else:
        return unif_probs(n)


def unif_histogram(n, **kwargs):
    return Histogram(0.0, 1.0, n, unif_probs(n))


def alt_histogram(n, consec=1, **kwargs):
    return Histogram(0.0, 1.0, n, alt_probs(n, consec))


def wobbly_histogram(n, consec=1, **kwargs):
    return Histogram(0.0, 1.0, n, wobbly_probs(n, consec))


def test_init_guard_clauses():
    with pytest.raises(TypeError):
        Histogram(vmin="str", vmax=1.0, num_atoms=10, probs=unif_probs(10))
    with pytest.raises(TypeError):
        Histogram(vmin=0.0, vmax="str", num_atoms=10, probs=unif_probs(10))
    with pytest.raises(TypeError):
        Histogram(vmin=0.0, vmax=1.0, num_atoms="str", probs=unif_probs(10))
    with pytest.raises(TypeError):
        Histogram(vmin=0.0, vmax=1.0, num_atoms=10, probs="str")
    with pytest.raises(TypeError):
        Histogram(vmin=0.0, vmax=1.0, num_atoms=10, probs=[0.1 for _ in range(10)])
    with pytest.raises(ValueError):
        Histogram(vmin=1.0, vmax=0.0, num_atoms=10, probs=unif_probs(10))
    with pytest.raises(ValueError):
        Histogram(vmin=0.0, vmax=1.0, num_atoms=0, probs=unif_probs(10))
    with pytest.raises(ValueError):
        Histogram(vmin=0.0, vmax=1.0, num_atoms=-1, probs=unif_probs(10))
    with pytest.raises(ValueError):
        Histogram(
            vmin=0.0,
            vmax=1.0,
            num_atoms=10,
            probs=np.array([0.1 for _ in range(10)])[None, ...],
        )
    with pytest.raises(ValueError):
        Histogram(
            vmin=0.0, vmax=1.0, num_atoms=10, probs=np.array([0.1 for _ in range(11)])
        )
    with pytest.raises(ValueError):
        Histogram(
            vmin=0.0, vmax=1.0, num_atoms=10, probs=np.array([-0.1 for _ in range(10)])
        )
    with pytest.raises(ValueError):
        Histogram(
            vmin=0.0, vmax=1.0, num_atoms=10, probs=np.array([0.05 for _ in range(10)])
        )


def test_empirical_guard_clauses():
    with pytest.raises(ValueError):
        unif_histogram(10).empirical(np.eye(2))
    with pytest.raises(TypeError):
        unif_histogram(10).empirical(np.array([0.5, 0.1, 0.8]), num_atoms=0.1)


def test_mixture_guard_clauses():
    with pytest.raises(TypeError):
        Histogram.mixture("str", [0.1, 0.9])
    with pytest.raises(TypeError):
        Histogram.mixture(["str", "str"], [0.1, 0.9])

    with pytest.raises(TypeError):
        Histogram.mixture([unif_histogram(2), alt_histogram(2)], "str")
    with pytest.raises(TypeError):
        Histogram.mixture([unif_histogram(2), alt_histogram(2)], [0, 1])

    with pytest.raises(ValueError):
        Histogram.mixture([unif_histogram(2), alt_histogram(2)], [1.0])

    with pytest.raises(ValueError):
        Histogram.mixture([unif_histogram(2), alt_histogram(2)], [0.1, 0.5])

    with pytest.raises(ValueError):
        Histogram.mixture([unif_histogram(2), alt_histogram(2)], [-0.1, 1.1])


def test_add_guard_clauses():
    with pytest.raises(TypeError):
        unif_histogram(10) + "str"
    with pytest.raises(TypeError):
        unif_histogram(10) + [0.1 for _ in range(10)]


def test_mul_guard_clauses():
    with pytest.raises(TypeError):
        unif_histogram(10) * "str"
    with pytest.raises(TypeError):
        unif_histogram(10) + [0.1 for _ in range(10)]
    with pytest.raises(TypeError):
        unif_histogram(10) * unif_histogram(10)


def test_sub_guard_clauses():
    with pytest.raises(TypeError):
        unif_histogram(10) - "str"
    with pytest.raises(TypeError):
        unif_histogram(10) - [0.1 for _ in range(10)]


def test_cdf_guard_clauses():
    with pytest.raises(TypeError):
        unif_histogram(10).cdf("str")


def test_inverse_cdf_guard_clauses():
    with pytest.raises(TypeError):
        unif_histogram(10).inverse_cdf("str")
    with pytest.raises(ValueError):
        unif_histogram(10).inverse_cdf(-0.1)
    with pytest.raises(ValueError):
        unif_histogram(10).inverse_cdf(1.1)


def test_sample_guard_clauses():
    with pytest.raises(TypeError):
        unif_histogram(10).sample("str", None)
    with pytest.raises(TypeError):
        unif_histogram(10).sample(100, "str")


def test_condition_guard_clauses():
    with pytest.raises(TypeError):
        unif_histogram(10).condition("str", 1.0)
    with pytest.raises(TypeError):
        unif_histogram(10).condition(0.0, "str")
    with pytest.raises(ValueError):
        unif_histogram(10).condition(2.0, -2.0)


def test_trim_guard_clauses():
    with pytest.raises(TypeError):
        unif_histogram(10).trim("str", 1.0)
    with pytest.raises(TypeError):
        unif_histogram(10).trim(0.0, "str")
    with pytest.raises(ValueError):
        unif_histogram(10).trim(2.0, -2.0)

    """
    >>> probs = [0.0, 0.0, 0.3, 0.7, 0.0, 0.0]
    >>> atoms = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    >>> bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    >>>
    >>> chop_start = bisect.bisect_left(bins[0:-1], 1.0)
    >>> chop_start
    2
    >>> chop_end = bisect.bisect_right(bins[1:], 4.0)
    >>> chop_end
    4
    """
    h = Histogram(
        vmin=-0.5,
        vmax=5.5,
        num_atoms=6,
        probs=np.array([0.0, 0.0, 0.3, 0.7, 0.0, 0.0]),
    )
    with pytest.raises(ValueError):
        h.trim(2.0, 5.5)
    with pytest.raises(ValueError):
        h.trim(-0.5, 3.0)
    h.trim(1.0, 4.0)  # should pass wo error


def test_pad_guard_clauses():
    with pytest.raises(TypeError):
        unif_histogram(10).pad("str", 1.0)
    with pytest.raises(TypeError):
        unif_histogram(10).pad(0.0, "str")
    with pytest.raises(ValueError):
        unif_histogram(10).pad(2.0, -2.0)
    with pytest.raises(ValueError):
        unif_histogram(10).pad(0.5, 1.0)
    with pytest.raises(ValueError):
        unif_histogram(10).pad(0.0, 0.5)


def test_shift_guard_clauses():
    with pytest.raises(TypeError):
        unif_histogram(10).shift(shift="str")
    with pytest.raises(TypeError):
        unif_histogram(10).shift(shift=unif_histogram(10))


def test_convolve_guard_clauses():
    with pytest.raises(TypeError):
        unif_histogram(10).convolve(other="str")
    with pytest.raises(TypeError):
        unif_histogram(10).convolve(other=1)
    with pytest.raises(TypeError):
        unif_histogram(10).convolve(other=1.0)
    with pytest.raises(ValueError):
        Histogram(0.0, 1.0, 10, unif_probs(10)).convolve(
            Histogram(-1.0, 1.0, 10, unif_probs(10))
        )
    with pytest.raises(ValueError):
        Histogram(0.0, 1.0, 10, unif_probs(10)).convolve(
            Histogram(0.0, 2.0, 10, unif_probs(10))
        )
    with pytest.raises(ValueError):
        Histogram(0.0, 1.0, 10, unif_probs(10)).convolve(
            Histogram(0.0, 1.0, 11, unif_probs(11))
        )


def test_convolve_slow_guard_clauses():
    with pytest.raises(TypeError):
        unif_histogram(10).convolve_slow(other="str")
    with pytest.raises(TypeError):
        unif_histogram(10).convolve_slow(other=1)
    with pytest.raises(TypeError):
        unif_histogram(10).convolve_slow(other=1.0)
    with pytest.raises(ValueError):
        Histogram(0.0, 1.0, 10, unif_probs(10)).convolve_slow(
            Histogram(-1.0, 1.0, 10, unif_probs(10))
        )
    with pytest.raises(ValueError):
        Histogram(0.0, 1.0, 10, unif_probs(10)).convolve_slow(
            Histogram(0.0, 2.0, 10, unif_probs(10))
        )
    with pytest.raises(ValueError):
        Histogram(0.0, 1.0, 10, unif_probs(10)).convolve_slow(
            Histogram(0.0, 1.0, 11, unif_probs(11))
        )


def test_renormalize_guard_clauses():
    with pytest.raises(ValueError):
        Histogram.renormalize(unif_histogram(10).probs * -1)


def test_mix_guard_clauses():
    with pytest.raises(TypeError):
        Histogram._mix("str", [0.1, 0.9])
    with pytest.raises(TypeError):
        Histogram._mix(["str", "str"], [0.1, 0.9])

    with pytest.raises(TypeError):
        Histogram._mix([unif_histogram(2), alt_histogram(2)], "str")
    with pytest.raises(TypeError):
        Histogram._mix([unif_histogram(2), alt_histogram(2)], [0, 1])

    with pytest.raises(ValueError):
        Histogram._mix([unif_histogram(2), alt_histogram(2)], [1.0])

    with pytest.raises(ValueError):
        Histogram._mix([unif_histogram(2), alt_histogram(2)], [0.1, 0.5])

    with pytest.raises(ValueError):
        Histogram._mix([unif_histogram(2), alt_histogram(2)], [-0.1, 1.1])

    with pytest.raises(ValueError):
        Histogram._mix(
            [
                Histogram(-1.0, 1.0, 2, unif_probs(2)),
                Histogram(0.0, 1.0, 2, unif_probs(2)),
            ],
            [0.1, 0.9],
        )
    with pytest.raises(ValueError):
        Histogram._mix(
            [
                Histogram(0.0, 1.0, 2, unif_probs(2)),
                Histogram(0.0, 10.0, 2, unif_probs(2)),
            ],
            [0.1, 0.9],
        )
    with pytest.raises(ValueError):
        Histogram._mix(
            [
                Histogram(0.0, 1.0, 2, unif_probs(2)),
                Histogram(0.0, 1.0, 10, unif_probs(10)),
            ],
            [0.1, 0.9],
        )


def test_empirical_return():
    h = Histogram.empirical(np.array([0.0, 0.3, 1.0]), num_atoms=2)
    np.testing.assert_allclose(h.probs, np.array([2 / 3, 1 / 3]), **TOLS)

    h = Histogram.empirical(np.array([0.0, 0.7, 1.0]), num_atoms=2)
    np.testing.assert_allclose(h.probs, np.array([1 / 3, 2 / 3]), **TOLS)


def test_mixture_return():
    h = Histogram.mixture([unif_histogram(2), alt_histogram(2)], [0.5, 0.5])
    assert type(h) == Histogram


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 10, 100])
def test_expectation_return(n):
    h = unif_histogram(n)
    np.testing.assert_allclose(h.expectation, 0.5, **TOLS)


def test_variance_return():
    h = Histogram(-2, 2, 2, probs=unif_probs(2))  # atoms on -1, 1
    np.testing.assert_allclose(h.variance, 1.0, **TOLS)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 10, 100])
def test_median_return(n):
    h = unif_histogram(n)
    np.testing.assert_allclose(h.median, 0.5, **TOLS)


def test_mode_return():
    h = Histogram(-2, 2, 2, probs=np.array([0.0, 1.0]))  # atoms on -1, 1
    np.testing.assert_allclose(h.mode, 1.0, **TOLS)


def test_differential_entropy_return():
    rng = np.random.default_rng()
    h = Histogram.empirical(rng.normal(size=[100_000]))
    np.testing.assert_allclose(
        h.differential_entropy, 0.5 * np.log(2 * 3.14 * 2.718), atol=1e-2, rtol=1e-2
    )
    h2 = h * 1000
    np.testing.assert_allclose(
        h2.differential_entropy,
        0.5 * np.log(2 * 3.14 * 2.718 * 1000**2),
        atol=1e-2,
        rtol=1e-2,
    )
    h3 = h * 0.001
    np.testing.assert_allclose(
        h3.differential_entropy,
        0.5 * np.log(2 * 3.14 * 2.718 * 0.001**2),
        atol=1e-2,
        rtol=1e-2,
    )


def test_eq_return():
    h = unif_histogram(10)

    h2 = unif_histogram(10)
    assert h == h2
    h3 = unif_histogram(5)
    assert h != h3
    h4 = alt_histogram(10)
    assert h != h4
    h5 = wobbly_histogram(10)
    assert h != h5


def test_repr_return():
    h = unif_histogram(5)
    assert type(h.__repr__()) == str


def test_add_return():
    assert type(unif_histogram(10) + 1) == Histogram
    assert type(unif_histogram(10) + 1.0) == Histogram
    assert type(unif_histogram(10) + unif_histogram(10)) == Histogram


def test_mul_return():
    assert type(unif_histogram(10) * 2) == Histogram
    assert type(unif_histogram(10) * 2.0) == Histogram


def test_neg_return():
    h = unif_histogram(10)
    assert type(-h) == Histogram
    np.testing.assert_allclose((-h).probs, (h * -1).probs, **TOLS)


def test_sub_return():
    assert type(unif_histogram(10) - 1) == Histogram
    assert type(unif_histogram(10) - 1.0) == Histogram

    h = Histogram(-1.0, 1.0, 10, probs=unif_probs(10))
    assert type(h - h) == Histogram  # these are independent copies of the rv, fyi
    np.testing.assert_allclose((h - h).probs, (h + h * -1).probs, **TOLS)
    np.testing.assert_allclose((h - h).probs, (h + -h).probs, **TOLS)


def test_radd_return():
    h = unif_histogram(10)
    assert type(1 + h) == Histogram
    np.testing.assert_allclose((1 + h).atoms, (h + 1).atoms)


def test_rmul_return():
    h = unif_histogram(10)
    assert type(2 * h) == Histogram
    np.testing.assert_allclose((2 * h).atoms, (h * 2).atoms)


def test_rsub_return():
    h = unif_histogram(10)
    assert type(1 - h) == Histogram
    np.testing.assert_allclose((1 - h).atoms, (-h + 1).atoms)
    np.testing.assert_allclose((1 - h).atoms, (-1 * h + 1).atoms)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 10, 100])
def test_cdf_return(n):
    h = unif_histogram(n)
    assert type(h.cdf(0.5)) == float
    np.testing.assert_allclose(h.cdf(-100.0), 0.0, **TOLS)
    np.testing.assert_allclose(h.cdf(0.0), 0.0, **TOLS)
    np.testing.assert_allclose(h.cdf(0.25), 0.25, **TOLS)
    np.testing.assert_allclose(h.cdf(0.5), 0.5, **TOLS)
    np.testing.assert_allclose(h.cdf(0.75), 0.75, **TOLS)
    np.testing.assert_allclose(h.cdf(1.0), 1.0, **TOLS)
    np.testing.assert_allclose(h.cdf(100.0), 1.0, **TOLS)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 10, 100])
def test_inverse_cdf_return(n):
    h = unif_histogram(n)
    assert type(h.inverse_cdf(0.5)) == float
    np.testing.assert_allclose(h.inverse_cdf(0.0), 0.0, **TOLS)
    np.testing.assert_allclose(h.inverse_cdf(0.25), 0.25, **TOLS)
    np.testing.assert_allclose(h.inverse_cdf(0.5), 0.5, **TOLS)
    np.testing.assert_allclose(h.inverse_cdf(0.75), 0.75, **TOLS)
    np.testing.assert_allclose(h.inverse_cdf(1.0), 1.0, **TOLS)


@pytest.mark.parametrize("hist", [unif_histogram, wobbly_histogram])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 10, 100])
def test_inverse_cdf_return_roundtrip(hist, n):
    h = hist(n)

    np.testing.assert_allclose(h.inverse_cdf(h.cdf(0.1)), 0.1, **TOLS)
    np.testing.assert_allclose(h.inverse_cdf(h.cdf(0.22)), 0.22, **TOLS)
    np.testing.assert_allclose(h.inverse_cdf(h.cdf(0.37)), 0.37, **TOLS)
    np.testing.assert_allclose(h.inverse_cdf(h.cdf(0.59)), 0.59, **TOLS)
    np.testing.assert_allclose(h.inverse_cdf(h.cdf(0.74)), 0.74, **TOLS)

    np.testing.assert_allclose(h.cdf(h.inverse_cdf(0.1)), 0.1, **TOLS)
    np.testing.assert_allclose(h.cdf(h.inverse_cdf(0.22)), 0.22, **TOLS)
    np.testing.assert_allclose(h.cdf(h.inverse_cdf(0.37)), 0.37, **TOLS)
    np.testing.assert_allclose(h.cdf(h.inverse_cdf(0.59)), 0.59, **TOLS)
    np.testing.assert_allclose(h.cdf(h.inverse_cdf(0.74)), 0.74, **TOLS)


@pytest.mark.parametrize("samples", [1, 2, 3, 4, 5, 6, 10, 100])
def test_sample_return(samples):
    h = unif_histogram(10)
    assert type(h.sample(1, None)) == np.ndarray
    assert h.sample(samples, None).shape == (samples,)


def test_condition():
    h = unif_histogram(2)
    hc1 = h.condition(-float("inf"), 0.5)
    np.testing.assert_allclose(hc1.probs, np.array([1.0, 0.0]), **TOLS)
    hc2 = h.condition(0.5, float("inf"))
    np.testing.assert_allclose(hc2.probs, np.array([0.0, 1.0]), **TOLS)

    h = unif_histogram(3)
    hc1 = h.condition(-float("inf"), 0.5)
    np.testing.assert_allclose(hc1.probs, np.array([2 / 3, 1 / 3, 0 / 3]), **TOLS)
    hc2 = h.condition(0.5, float("inf"))
    np.testing.assert_allclose(hc2.probs, np.array([0 / 3, 1 / 3, 2 / 3]), **TOLS)

    h = unif_histogram(4)
    hc1 = h.condition(0.25, 0.75)
    np.testing.assert_allclose(
        hc1.probs, np.array([0 / 4, 2 / 4, 2 / 4, 0 / 4]), **TOLS
    )
    hc2 = h.condition(0.125, 0.875)
    np.testing.assert_allclose(
        hc2.probs, np.array([1 / 6, 2 / 6, 2 / 6, 1 / 6]), **TOLS
    )


@pytest.mark.parametrize("hist", [unif_histogram, alt_histogram])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 10, 100])
def test_pad(hist, n):
    h = hist(n)

    hp = h.pad(-1.0, 1.0)
    np.testing.assert_allclose(hp.probs[-n:], h.probs)
    np.testing.assert_allclose(hp.probs[:-n], 0.0)

    hp = h.pad(-1.0, 1.0, extra=True)
    np.testing.assert_allclose(hp.probs[-(n + 1) : -1], h.probs)
    np.testing.assert_allclose(hp.probs[: -(n + 1)], 0.0)

    hp = h.pad(0.0, 2.0)
    np.testing.assert_allclose(hp.probs[:n], h.probs)
    np.testing.assert_allclose(hp.probs[n:], 0.0)

    hp = h.pad(0.0, 2.0, extra=True)
    np.testing.assert_allclose(hp.probs[1 : (n + 1)], h.probs)
    np.testing.assert_allclose(hp.probs[(n + 1) :], 0.0)


@pytest.mark.parametrize("hist", [unif_histogram, alt_histogram])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_trim(hist, n):
    # numerics not so good for large n = 10, 100,
    # sometimes the bins are like -2e-16
    # when they should be zero, and the left bin is dropped on trim.

    h = hist(n)
    hp = h.pad(-1.0, 1.0)
    print(hp.probs)
    print(hp.bin_edges)
    print(hp.atom_stride)
    ht = hp.trim(0.0, 1.0)
    assert h == ht


@pytest.mark.parametrize("hist, probs", [(unif_histogram, unif_probs)])
@pytest.mark.parametrize("start_n, end_n", [(2, 3), (3, 2)])
def test_rebin_unaligned_unif(hist, probs, start_n, end_n):
    h = hist(start_n)
    h = h.rebin(0.0, 1.0, end_n)
    np.testing.assert_allclose(h.probs, probs(end_n), **TOLS)


def test_rebin_unaligned_alt():
    h = alt_histogram(2)
    h = h.rebin(0.0, 1.0, 3)
    np.testing.assert_allclose(h.probs, np.array([2 / 3, 1 / 3, 0 / 3]), **TOLS)

    h = alt_histogram(3)
    h = h.rebin(0.0, 1.0, 2)
    np.testing.assert_allclose(h.probs, np.array([1 / 2, 1 / 2]), **TOLS)


@pytest.mark.parametrize("hist, probs", [(unif_histogram, unif_probs)])
@pytest.mark.parametrize("start_n, end_n", [(10, 100), (100, 10)])
def test_rebin_aligned_unif(hist, probs, start_n, end_n):
    h = hist(start_n)
    h = h.rebin(0.0, 1.0, end_n)
    np.testing.assert_allclose(h.probs, probs(end_n), **TOLS)


@pytest.mark.parametrize("hist, probs", [(alt_histogram, alt_probs)])
@pytest.mark.parametrize("start_n, end_n", [(10, 20), (10, 100), (20, 10), (100, 10)])
def test_rebin_aligned_alt(hist, probs, start_n, end_n):
    h = hist(start_n)
    h = h.rebin(0.0, 1.0, end_n)
    np.testing.assert_allclose(h.probs, probs(end_n, end_n // start_n), **TOLS)


@pytest.mark.parametrize("hist", [unif_histogram, alt_histogram, wobbly_histogram])
@pytest.mark.parametrize("start_n, end_n", [(10, 20), (10, 100), (20, 10), (100, 10)])
def test_rebin_idempotent(hist, start_n, end_n):
    h = hist(start_n)
    h = h.rebin(0.0, 1.0, end_n)
    h2 = h.rebin(0.0, 1.0, end_n)
    assert h2 == h


def test_shift_return():
    assert type(unif_histogram(10).shift(1)) == Histogram
    assert type(unif_histogram(10).shift(1.0)) == Histogram


@pytest.mark.parametrize("hist", [unif_histogram, alt_histogram])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 10, 100])
def test_convolve(hist, n):
    h = hist(n)
    hc = h.convolve(h)
    hcs = h.convolve_slow(h)
    np.testing.assert_allclose(hc.probs, hcs.probs, **TOLS)


def test_mix():
    h1 = Histogram(0.0, 1.0, 2, np.array([0.5, 0.5]))
    h2 = Histogram(0.0, 1.0, 2, np.array([0.5, 0.5]))
    assert Histogram._mix([h1, h2], [0.2, 0.8]) == h1

    h3 = Histogram(0.0, 1.0, 2, np.array([1.0, 0.0]))
    assert Histogram._mix([h2, h3], [0.8, 0.2]) == Histogram(
        0.0, 1.0, 2, np.array([0.6, 0.4])
    )
