import bisect
import logging
import math
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np


class Histogram:
    def __init__(
        self,
        vmin: Union[int, float],
        vmax: Union[int, float],
        num_atoms: int,
        probs: np.ndarray,
    ) -> None:
        """Histogram class to represent the distribution of a one-dimensional random variable.

        Args:
            vmin: Minimum permitted value for the histogram's random variable.
            vmax: Maximum permitted value for the histogram's random variable.
            num_atoms: Number of bins for the histogram.
            probs: Probabilities for the bins in the histogram.

        Returns:
            A new Histogram instance representing the distribution of the new variable.

        Raises:
            TypeError: If vmin is not an int or float.
            TypeError: If vmax is not an int or float.
            TypeError: If num_atoms is not an int.
            TypeError: If probs is not a numpy.ndarray.
            ValueError: If vmin >= vmax.
            ValueError: If num_atoms <= 0.
            ValueError: If probs.shape is not equal to (num_atoms,).
            ValueError: If probs contains negative values.
            ValueError: If probs does not sum to one.
        """
        if not isinstance(vmin, int) and not isinstance(vmin, float):
            raise TypeError("input 'vmin' must be int or float type.")
        if not isinstance(vmax, int) and not isinstance(vmax, float):
            raise TypeError("input 'vmax' must be int or float type.")
        if not isinstance(num_atoms, int):
            raise TypeError("input 'num_atoms' must be int type.")
        if not isinstance(probs, np.ndarray):
            raise TypeError("input 'probs' must be numpy.ndarray type.")
        if vmin >= vmax:
            raise ValueError("input 'vmin' must be less than 'vmax'.")
        if num_atoms <= 0:
            raise ValueError("input 'num_atoms' must be positive.")
        if len(probs.shape) != 1 or probs.shape[0] != num_atoms:
            raise ValueError("input 'probs' must be of shape (num_atoms,).")
        if not np.allclose(probs, np.abs(probs)):
            raise ValueError("input 'probs' must be non-negative.")
        if not np.allclose(np.sum(probs), 1.0):
            raise ValueError("input 'probs' must sum to one.")

        self._vmin = vmin
        self._vmax = vmax
        self._num_atoms = num_atoms
        self._probs = probs

    @classmethod
    def empirical(cls, vs: np.ndarray, num_atoms: Optional[int] = None) -> "Histogram":
        """Create a histogram instance fit to the data.

        Args:
            vs: A numpy.ndarray of shape (n,) containing the data.
            num_atoms: Optional number of bins. If None, uses sqrt(n).
                Default value is None.

        Returns:
            A new Histogram instance describing the data.

        Raises:
            ValueError: If data array is not one-dimensional.
            TypeError: If num_atoms is not int and not None.
        """
        if len(vs.shape) != 1:
            raise ValueError("Only 1-dimensional data is supported.")
        if not isinstance(num_atoms, int) and num_atoms is not None:
            raise TypeError("Input 'num_atoms' must be int or None.")
        if num_atoms is None:
            num_atoms = math.ceil(vs.shape[0] ** 0.5)

        vmin = np.min(vs, axis=-1)
        vmax = np.max(vs, axis=-1)
        atom_stride = (vmax - vmin) / num_atoms

        idxs = np.floor((vs - vmin) / atom_stride).astype(np.int32)
        idxs = np.minimum(np.maximum(idxs, 0), num_atoms - 1)
        eye = np.eye(num_atoms)
        idxs_onehot = np.take_along_axis(eye, idxs[..., None], axis=0)
        counts = np.sum(idxs_onehot, axis=0)
        probs = Histogram.renormalize(counts)

        return Histogram(
            vmin=vmin,
            vmax=vmax,
            num_atoms=num_atoms,
            probs=probs,
        )

    @classmethod
    def mixture(cls, hists: List["Histogram"], weights: List[float]) -> "Histogram":
        f"""Create a mixture distribution from clusters and weights.

        In detail, this creates a histogram for a random variable of the form
        ```Y = i1 * X1 + ... + iN * XN```
        where only one of ```i1, ..., iN``` is 1 and the rest are zero,
        and the probability that ```ik == 1``` is ```weights[k]```,
        and where the distribution of each ```Xk``` is modeled by ```hists[k]```,
        and is independent of i1, ..., iN.

        Args:
            hists: List of Histograms serving as clusters for the mixture.
                The Histograms are rebinned automatically to enable their mixture.
            weights: List of floats serving as weights for the mixture.

        Returns:
            A new Histogram representing the mixture distribution.

        Raises:
            TypeError: If hists is not a list of Histograms.
            TypeError: If weights is not a list of floats.
            ValueError: If len(hists) != len(weights).
            ValueError: If sum(weights) != 1.0.
            ValueError: If min(weights) < 0.0.
        """
        if not isinstance(hists, list) or not isinstance(hists[0], Histogram):
            raise TypeError("input 'hists' must be a list of Histograms.")
        if not isinstance(weights, list) or not isinstance(weights[0], float):
            raise TypeError("input 'weights' must be a list of floats.")
        if len(hists) != len(weights):
            raise ValueError("inputs 'hists' and 'weight' must have same length.")
        if not np.allclose(sum(weights), 1.0):
            raise ValueError("input 'weights' must sum to one.")
        if min(weights) < 0.0:
            raise ValueError("input 'weights' must be all non-negative.")

        hists = [h.trim() for h in hists]
        new_vmin = min(h.vmin for h in hists)
        new_vmax = max(h.vmax for h in hists)
        new_num_atoms = math.ceil(sum(h.num_atoms**2 for h in hists) ** 0.5)
        spec = dict(
            new_vmin=new_vmin,
            new_vmax=new_vmax,
            new_num_atoms=new_num_atoms,
        )
        hists = [h.rebin(**spec) for h in hists]
        return Histogram._mix(hists, weights)

    @property
    def vmin(self):
        """float: Minimum permitted value for the histogram's random variable."""
        return self._vmin

    @property
    def vmax(self):
        """float: Maximum permitted value for the histogram's random variable."""
        return self._vmax

    @property
    def num_atoms(self):
        """int: Number of bins for the histogram."""
        return self._num_atoms

    @property
    def probs(self):
        """numpy.ndarray: A copy of the probabilities for the bins in the histogram."""
        return np.copy(self._probs)

    @property
    def atom_stride(self) -> float:
        """float: The histogram bin width."""
        return (self.vmax - self.vmin) / self.num_atoms  # num atoms = num bins

    @property
    def atom_min(self) -> float:
        """float: The center of the leftmost histogram bin."""
        return self.vmin + self.atom_stride / 2

    @property
    def atom_max(self) -> float:
        """float: The center of the rightmost histogram bin."""
        return self.vmax - self.atom_stride / 2

    @property
    def atoms(self) -> np.ndarray:
        """numpy.ndarray: The centers for the histogram bins."""
        output = np.arange(self.num_atoms) * self.atom_stride + self.atom_min
        np.testing.assert_allclose(output[-1], self.atom_max)
        return output

    @property
    def bin_edges(self) -> np.ndarray:
        """numpy.ndarray: The edges for the histogram bins."""
        output = np.arange(self.num_atoms + 1) * self.atom_stride + self.vmin
        np.testing.assert_allclose(output[-1], self.vmax)
        return output

    @property
    def support(self) -> List[int]:
        """List[int]: Lists the bin indices with nonzero probability."""
        ls = []
        probs = self.probs
        for i in range(self.num_atoms):
            if probs[i] > 0.0:
                ls.append(i)
        return ls

    @property
    def extrema(self) -> Tuple[float, float]:
        """Tuple[float, float]: The left edge of the leftmost bin with nonzero probability mass,
        and the right edge of the rightmost bin with nonzero probability mass.
        """
        supp = self.support
        edges = self.bin_edges
        return edges[0:-1][supp[0]], edges[1:][supp[-1]]

    @property
    def expectation(self) -> float:
        """float: The expectation (mean) of the histogram."""
        return np.sum(self.atoms * self.probs, axis=-1)

    @property
    def variance(self) -> float:
        """float: The variance of the histogram."""
        mu = self.expectation
        return np.sum(np.square(self.atoms - mu) * self.probs, axis=-1)

    @property
    def median(self) -> float:
        """float: The median of the histogram, alias for self.inverse_cdf(0.5)."""
        return self.inverse_cdf(0.5)

    @property
    def mode(self) -> float:
        """float: The mode of the histogram."""
        return self.atoms[np.argmax(self.probs, axis=-1)].item()

    @property
    def differential_entropy(self) -> float:
        """float: The differential entropy of the underlying density, in nats."""
        density = self.probs / self.atom_stride  # integral of density over bin = prob
        return -np.sum(
            self.atom_stride * density * np.log(density + 1e-6), axis=-1
        ).item()

    def __eq__(
        self, other: "Histogram", atol: float = 1e-4, rtol: float = 1e-4
    ) -> bool:
        """Determines whether two histograms are equal, up to numerical errors.

        To use non-default atol and rtol, must use the ```__eq__``` form,
        not the ```==``` operator.

        Args:
            other: Another Histogram instance.
            atol: Absolute error tolerance for probability comparison.
            rtol: Relative error tolerance for probability comparison.

        Returns:
            True if both instances are equal, up to numerical errors.
            False otherwise.
        """
        if not np.allclose(self.vmin, other.vmin):
            return False
        if not np.allclose(self.vmax, other.vmax):
            return False
        if self.num_atoms != other.num_atoms:
            return False
        if not np.allclose(self.probs, other.probs, atol=atol, rtol=rtol):
            return False
        return True

    def __repr__(self) -> str:
        """A string representation of the Histogram object.

        Returns:
            A string containing information about vmin, vmax, num_atoms,
                and the ```__repr__``` of self.probs.
        """
        ls = []
        ls.append("Histogram(\n")
        ls.append(f"    vmin={self.vmin},\n")
        ls.append(f"    vmax={self.vmax},\n")
        ls.append(f"    num_atoms={self.num_atoms},\n")
        ls.append(f"    probs={self.probs.__repr__()},\n")
        ls.append(")")
        return "".join(ls)

    def __add__(self, other: Union[int, float, "Histogram"]) -> "Histogram":
        """Adds a scalar or an independent random variable to the current histogram's random variable.

        Args:
            other: An int, float, or Histogram instance.

        Returns:
            A new Histogram instance representing the distribution of the new variable.

        Raises:
            TypeError: If other is not an int, float, or Histogram.
            ValueError: If other is a Histogram with different bins than self.
        """
        if isinstance(other, int) or isinstance(other, float):
            return self.shift(other)
        if isinstance(other, Histogram):
            h1 = self.trim()
            h2 = other.trim()
            spec = dict(
                new_vmin=min(self.vmin, other.vmin),
                new_vmax=max(self.vmax, other.vmax),
                new_num_atoms=math.ceil(
                    (self.num_atoms**2 + other.num_atoms**2) ** 0.5
                ),
            )
            h1 = h1.rebin(**spec)
            h2 = h2.rebin(**spec)
            return h1.convolve(h2)
        raise TypeError("input 'other' must be int, float, or Histogram type.")

    def __mul__(self, other: Union[int, float]) -> "Histogram":
        """Multiplies the current histogram's random variable by a scalar.

        Args:
            other: An int or float.

        Returns:
            A new Histogram instance representing the distribution of the new variable.

        Raises:
            TypeError: If other is not an int or float.
        """
        if not isinstance(other, int) and not isinstance(other, float):
            raise TypeError("input 'coef' must be int or float type.")
        return Histogram(
            vmin=min(other * self.vmin, other * self.vmax),
            vmax=max(other * self.vmin, other * self.vmax),
            num_atoms=self.num_atoms,
            probs=self.probs if other >= 0 else self.probs[::-1],
        )

    def __neg__(self) -> "Histogram":
        """
        Negation of the histogram's random variable.

        Returns:
            A new Histogram instance representing the distribution of the new variable.
        """
        return self.__mul__(-1)

    def __sub__(self, other: Union[int, float, "Histogram"]) -> "Histogram":
        """Subtracts a scalar or an independent random variable from the current histogram's random variable.

        Args:
            other: An int, float, or Histogram instance.

        Returns:
            A new Histogram instance representing the distribution of the new variable.

        Raises:
            TypeError: If other is not an int, float, or Histogram.
            ValueError: If other is a Histogram with different bins than self.
        """
        if (
            isinstance(other, int)
            or isinstance(other, float)
            or isinstance(other, Histogram)
        ):
            return self.__add__(-other)
        raise TypeError("input 'other' must be int, float, or Histogram type.")

    def __radd__(self, other: Union[int, float, "Histogram"]) -> "Histogram":
        """Adds a scalar or an independent random variable to the current histogram's random variable.

        Args:
            other: An int, float, or Histogram instance.

        Returns:
            A new Histogram instance representing the distribution of the new variable.

        Raises:
            TypeError: If other is not an int, float, or Histogram.
            ValueError: If other is a Histogram with different bins than self.
        """
        return self.__add__(other)

    def __rmul__(self, other: Union[int, float]) -> "Histogram":
        """Multiplies the current histogram's random variable by a scalar.

        Args:
            other: An int or float.

        Returns:
            A new Histogram instance representing the distribution of the new variable.

        Raises:
            TypeError: If other is not an int or float.
        """
        return self.__mul__(other)

    def __rsub__(self, other: Union[int, float, "Histogram"]) -> "Histogram":
        """Subtracts the current histogram's random variable from a scalar or an independent random variable.

        Args:
            other: An int, float, or Histogram instance to be subtracted from.

        Returns:
            A new Histogram instance representing the distribution of the new variable.

        Raises:
            TypeError: If other is not an int, float, or Histogram.
            ValueError: If other is a Histogram with different bins than self.
        """
        return self.__sub__(other).__mul__(-1)

    def plot(self) -> None:
        """Plot the histogram using matplotlib."""
        plt.bar(
            self.atoms,
            self.probs,
            width=self.atom_stride,
            edgecolor="black",
            align="center",
        )
        plt.show()

    def cdf(self, v: float) -> float:
        """Evaluate the cumulative distribution function at particular point.

        Evaluating at points between histogram bin edges will include a fraction of the containing bin's probability mass.
        Points less than self.vmin or more than self.vmax are allowed, and return 0.0 or 1.0, respectively.

        Args:
            v: The point to evaluate the CDF at.

        Returns:
            The probability of a sample being less than or equal to v.

        Raises:
            TypeError: if v is not an int or float.
        """
        if not isinstance(v, int) and not isinstance(v, float):
            raise TypeError("Input 'v' must be of type int or float.")

        if v < self.vmin:
            return 0.0
        if self.vmax <= v:
            return 1.0

        probs = self.probs
        edges = self.bin_edges
        for i in range(0, self.num_atoms):
            if edges[i] <= v < edges[i + 1]:
                break

        prob = 0.0
        if i > 0:
            prob += np.sum(probs[0:i], axis=-1)
        prob += ((v - edges[i]) / self.atom_stride) * probs[i]
        return prob.item()

    def inverse_cdf(self, p: float) -> float:
        """Evaluate the inverse of the cumulative distribution function at a particular point.

        Implemented so that if self.probs is all nonzero, self.inverse_cdf(self.cdf(v)) == v
        for any v between self.vmin and self.vmax. This means probability values between
        bin increments are assigned to sample space based on their distance to the increments.

        Args:
            p: A probability value with 0 <= p <= 1.

        Returns:
            The value in the sample space corresponding to the cumulative probability given.

        Raises:
            TypeError: If p is not an int or float type.
            ValueError: If p is not between 0 and 1, inclusive.
        """
        if not isinstance(p, int) and not isinstance(p, float):
            raise TypeError("Input 'p' must be of type int or float.")
        if not (0.0 <= p <= 1.0):
            raise ValueError("Input 'p' must be in range 0 <= p <= 1.")

        if p == 0.0:
            return self.vmin
        if p == 1.0:
            return self.vmax

        atoms = self.atoms
        probs = self.probs
        summed = np.concatenate([np.array([0.0]), np.cumsum(probs, axis=-1)], axis=-1)
        for i in range(0, self.num_atoms):
            if summed[i] <= p < summed[i + 1]:
                break
        quantile = atoms[i] - self.atom_stride / 2
        quantile += ((p - summed[i]) / probs[i]) * self.atom_stride
        return quantile.item()

    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> float:
        """Sample n points from the distribution represented by the histogram.

        Args:
            n: The number of points to sample. Default value is 1.
            rng: An optional numpy.random.Generator object to support the modern numpy RNG API.
                 If omitted, uses numpy's legacy global RNG. Default value is None.

        Returns:
            A numpy.ndarray with shape (n,) containing the samples.

        Raises:
            TypeError: If input n is not an int.
            TypeError: If input rng is not numpy.random.Generator or None.
        """
        if not isinstance(n, int):
            raise TypeError("Input 'n' must be an int.")
        if not isinstance(rng, np.random.Generator) and rng is not None:
            raise TypeError(
                "Input 'rng' must be numpy.random.Generator or None (for legacy rng)."
            )
        if rng is None:
            u = np.random.uniform(low=0.0, high=1.0, size=[n])
        else:
            u = rng.uniform(low=0.0, high=1.0, size=[n])
        return np.vectorize(self.inverse_cdf)(u)

    def condition(
        self, left: float = -float("inf"), right: float = float("inf")
    ) -> "Histogram":
        """Conditions the random variable as being in an interval (left, right).

        Internally, this method zeros out probability mass outside the range and renormalizes.
        If the interval divides a bin, the corresponding fraction of the bin's probability mass will kept.

        Args:
            left: Low bound for the random variable. Default value -float('inf').
            right: High bound for the random variable. Default value float('inf').

        Return:
            New Histogram representing conditional distribution of random variable.
        """
        if not isinstance(left, int) and not isinstance(left, float):
            raise TypeError("input 'left' must be int or float type.")
        if not isinstance(right, int) and not isinstance(right, float):
            raise TypeError("input 'right' must be int or float type.")
        if left >= right:
            raise ValueError("input 'left' must be less than 'right'.")

        edges = self.bin_edges
        probs = self.probs  # it's a deep copy
        left_i = -float("inf")
        right_i = float("inf")
        for i in range(0, self.num_atoms):
            if edges[i] <= left < edges[i + 1]:
                left_i = i
            if edges[i] <= right < edges[i + 1]:
                right_i = i

        if left_i == right_i:
            probs = np.zeros_like(probs)
            probs[left_i] = 1.0
        else:
            for i in range(self.num_atoms):
                if i < left_i or right_i < i:
                    probs[i] = 0.0
                if i == left_i:
                    probs[i] *= (edges[i + 1] - left) / self.atom_stride
                if i == right_i:
                    probs[i] *= (right - edges[i]) / self.atom_stride

        return Histogram(
            vmin=self.vmin,
            vmax=self.vmax,
            num_atoms=self.num_atoms,
            probs=Histogram.renormalize(probs),
        )

    def pad(
        self, left: Optional[float], right: Optional[float], extra: bool = False
    ) -> "Histogram":
        """Pad the histogram with zero-mass bins until the outer edges
        exceed the range given.

        Args:
            left: Left pad target. If None, uses self.vmin.
            right: Right pad target. If None, uses self.vmax.
            extra: Adds one extra atom to each side of the new histogram,
                beyond what is needed to cover the range specified. Defaults to False.

        Returns:
            New Histogram whose bins minimally contain the range [left, right].

        Raises:
            TypeError: If left is not int or float.
            TypeError: If right is not int or float.
            ValueError: If left >= right.
            ValueError: If self.vmin < left.
            ValueError: If right < self.vmax.
        """
        if (
            left is not None
            and not isinstance(left, int)
            and not isinstance(left, float)
        ):
            raise TypeError("input 'left' must be int or float.")
        if (
            right is not None
            and not isinstance(right, int)
            and not isinstance(right, float)
        ):
            raise TypeError("input 'right' must be int or float.")
        if left is not None and right is not None and left >= right:
            raise ValueError("input 'left' must be less than 'right'.")
        if left is None:
            left = self.vmin
        if right is None:
            right = self.vmax

        if self.vmin < left:
            raise ValueError(
                f"missing left bin coverage of old distribution: (old_vmin < left), ({self.vmin} < {left})"
            )
        if right < self.vmax:
            raise ValueError(
                f"missing right bin coverage of old distribution: (right < old_vmax), ({right} < {self.vmax})"
            )

        atoms = self.atoms
        vmin = self.vmin
        vmax = self.vmax

        # pad old distribution with left atoms of zero probability mass
        # until old bins exceed left range of new ones
        left_atom_padding = []
        while left < vmin:
            vmin -= self.atom_stride
            left_atom_padding.insert(0, vmin + self.atom_stride / 2)
        if extra:
            # extra padding
            vmin -= self.atom_stride
            left_atom_padding.insert(0, vmin + self.atom_stride / 2)
        left_atom_padding = np.array(left_atom_padding)

        # pad old distribution with right atoms of zero probability mass
        # until old bins exceed right range of new ones
        right_atom_padding = []
        while vmax < right:
            vmax += self.atom_stride
            right_atom_padding.append(vmax - self.atom_stride / 2)
        if extra:
            # extra padding
            vmax += self.atom_stride
            right_atom_padding.append(vmax - self.atom_stride / 2)
        right_atom_padding = np.array(right_atom_padding)

        atoms = np.concatenate(
            [
                left_atom_padding,
                atoms,
                right_atom_padding,
            ],
            axis=0,
        )
        probs = np.concatenate(
            [
                np.zeros_like(left_atom_padding),
                self.probs,
                np.zeros_like(right_atom_padding),
            ],
            axis=0,
        )

        return Histogram(
            vmin=vmin,
            vmax=vmax,
            num_atoms=atoms.shape[0],
            probs=probs,
        )

    def trim(
        self, left: Optional[float] = None, right: Optional[float] = None
    ) -> "Histogram":
        """Trim the histogram of zero-mass bins until the outer edges
        are contained within the range given.

        Calling ```trim``` after ```pad``` on a histogram with a large number of atoms
        relative to the range of values may cause numerical errors in bin edge calculation.
        In this case, you may need to expand ```left``` and ```right```
        by a small amount (e.g., less than the atom stride), versus the original
        vmin and vmax values used prior to padding, to avoid triggering an error
        for slicing nonzero probability mass.

        Args:
            left: Left trim target. If None, uses self.extrema[0]. Default value None.
            right: Right trim target. If None, uses self.extrema[1]. Default value None.

        Returns:
            New Histogram instance whose bins maximally lie within [left, right].

        Raises:
            TypeError: If left is not int or float.
            TypeError: If right is not int or float.
            ValueError: If left >= right.
            ValueError: If nonzero probability mass is requested trimmed.
        """
        if (
            left is not None
            and not isinstance(left, int)
            and not isinstance(left, float)
        ):
            raise TypeError("input 'left' must be int or float.")
        if (
            right is not None
            and not isinstance(right, int)
            and not isinstance(right, float)
        ):
            raise TypeError("input 'right' must be int or float.")
        if left is not None and right is not None and left >= right:
            raise ValueError("input 'left' must be less than 'right'.")
        extrema = self.extrema
        if left is None:
            left = extrema[0]
        if right is None:
            right = extrema[1]
        logging.debug(f"left: {left}")
        logging.debug(f"right: {right}")

        chop_start = bisect.bisect_left(self.bin_edges[0:-1], left)
        chop_end = bisect.bisect_right(self.bin_edges[1:], right)
        logging.debug(f"chop_start: {chop_start}")
        logging.debug(f"chop_end: {chop_end}")

        if chop_start > 0 and max(self.probs[0:chop_start]) > 0:
            raise ValueError("input 'left' slices nonzero probability mass.")
        if chop_end < self.num_atoms and max(self.probs[chop_end:]) > 0:
            raise ValueError("input 'right' slices nonzero probability mass.")

        new_vmin = self.bin_edges[0:-1][chop_start]
        new_vmax = self.bin_edges[1:][chop_end - 1]
        if chop_end == self.num_atoms:
            chop_end = None
        new_num_atoms = self.atoms[chop_start:chop_end].shape[0]
        new_probs = self.probs[chop_start:chop_end]
        return Histogram(
            vmin=new_vmin,
            vmax=new_vmax,
            num_atoms=new_num_atoms,
            probs=new_probs,
        )

    def rebin(
        self,
        new_vmin: Optional[float],
        new_vmax: Optional[float],
        new_num_atoms: Optional[int],
    ) -> "Histogram":
        """Rebin the histogram.

        Implemented so that the probability mass of each old bin is shared according
        to the proportion of its intersection with each new bin.

        Args:
            new_vmin: Minimum permitted value for the new histogram's random variable.
                If None, uses current self.vmin.
            new_vmax: Maximum permitted value for the new histogram's random variable.
                If None, uses current self.vmax.
            new_num_atoms: Number of bins for the new histogram.
                If None, uses current self.num_atoms.

        Returns:
            A new Histogram instance with the rebinned probability mass.

        Raises:
            ValueError: If the new_vmin is larger than the old one.
            ValueError: If the new_vmax is smaller than the old one.
            RuntimeError: If the algorithm does not function as expected.
                This should never occur.
        """
        if new_vmin is None:
            new_vmin = self.vmin
        if new_vmax is None:
            new_vmax = self.vmax
        if new_num_atoms is None:
            new_num_atoms = self.num_atoms

        old_padded = self.pad(new_vmin, new_vmax, extra=True)
        old_probs = old_padded.probs
        old_edges = old_padded.bin_edges

        # loop over new bins, figure out how much old probability mass they intersect with
        new_atom_stride = (new_vmax - new_vmin) / new_num_atoms
        new_probs = np.zeros(dtype=self.probs.dtype, shape=[new_num_atoms])
        new_edges = np.arange(new_num_atoms + 1) * new_atom_stride + new_vmin
        j = 0
        for i in range(0, new_num_atoms):
            logging.debug(f"new i: {i}")
            # for each new i, we should only get to it once its left side is past
            # the left side of of old bin j.
            old_left = old_edges[j]
            old_right = old_edges[j + 1]
            new_left = new_edges[i]
            new_right = new_edges[i + 1]
            if not (old_left <= new_left):
                raise RuntimeError(f"got old_left > new_left: {old_left} > {new_left}")

            # now we loop over old bins j and add in the contrib to new bin i
            while old_left < new_right:
                logging.debug(f"current i,j: {i},{j}")
                old_left = old_edges[j]
                old_right = old_edges[j + 1]
                logging.debug(f"new_left, new_right: {new_left}, {new_right}")
                logging.debug(f"old_left, old_right: {old_left}, {old_right}")

                # compute intersection between current i and j
                intersect_left = max(new_left, old_left)
                intersect_right = min(new_right, old_right)
                frac = (intersect_right - intersect_left) / self.atom_stride
                new_probs[i] += frac * old_probs[j]
                # if current new bin's range extends beyond current old bin
                # continue to next old bin, otherwise we're done with this new bin
                if old_right < new_right:
                    j += 1
                else:
                    break

        return Histogram(
            vmin=new_vmin,
            vmax=new_vmax,
            num_atoms=new_num_atoms,
            probs=Histogram.renormalize(new_probs),
        )

    def shift(self, scalar: Union[int, float]) -> "Histogram":
        """Add a scalar to the histogram's random variable.

        For addition of independent random variables, see ```convolve``` or ```__add__```.

        Args:
            scalar: A scalar to be added.

        Returns:
            A new Histogram instance representing the shifted variable.

        Raises:
            TypeError: If the inputted shift variable is not an int or float.
        """
        if not isinstance(scalar, int) and not isinstance(scalar, float):
            raise TypeError("input 'scalar' must be int or float type.")
        return Histogram(
            vmin=scalar + self.vmin,
            vmax=scalar + self.vmax,
            num_atoms=self.num_atoms,
            probs=self.probs,  # probs is already a deep copy of self._probs
        )

    def convolve(self, other: "Histogram") -> "Histogram":
        """Convolve two Histogram instances in O(nlogn) time, where n is bin count,
        using the circular convolution theorem.

        For a simpler method to compare against, see ```convolve_slow```.

        Args:
            other: A Histogram instance to be convolved with.

        Returns:
            A new Histogram instance representing the addition of the random variables.

        Raises:
            TypeError: If the inputted 'other' variable is not a Histogram.
            ValueError: If self.bin_edges != other.bin_edges up to numerical precision.
        """
        if not isinstance(other, Histogram):
            raise TypeError("input 'other' must be Histogram type.")
        if not np.allclose(self.bin_edges, other.bin_edges, atol=1e-4, rtol=1e-4):
            raise ValueError("input other.bin_edges must match self.bin_edges.")

        conv_size = 2 * self.num_atoms - 1
        fx = np.fft.rfft(self.probs, n=conv_size)
        fy = np.fft.rfft(other.probs, n=conv_size)
        probs = np.fft.irfft(fx * fy, n=conv_size)
        return Histogram(
            vmin=self.atom_min + other.atom_min - self.atom_stride / 2,
            vmax=self.atom_max + other.atom_max + self.atom_stride / 2,
            num_atoms=2 * self.num_atoms - 1,
            probs=Histogram.renormalize(probs),
        )

    def convolve_slow(self, other: "Histogram") -> "Histogram":
        """Convolve two Histogram instances in O(n^2) time, where n is bin count.

        For a faster method, see ```convolve```.

        Args:
            other: A Histogram instance to be convolved with.

        Returns:
            A new Histogram instance representing the addition of the random variables.

        Raises:
            TypeError: If the inputted 'other' variable is not a Histogram.
            ValueError: If self.bin_edges != other.bin_edges up to numerical precision.
        """
        if not isinstance(other, Histogram):
            raise TypeError("input 'other' must be Histogram type.")
        if not np.allclose(self.bin_edges, other.bin_edges, atol=1e-4, rtol=1e-4):
            raise ValueError("input other.bin_edges must match self.bin_edges.")

        probs = np.zeros(dtype=self.probs.dtype, shape=[2 * self.num_atoms - 1])
        for i in range(0, len(self.atoms)):
            for j in range(0, len(self.atoms)):
                probs[i + j] += self.probs[i] * other.probs[j]
        return Histogram(
            vmin=self.atom_min + other.atom_min - self.atom_stride / 2,
            vmax=self.atom_max + other.atom_max + self.atom_stride / 2,
            num_atoms=2 * self.num_atoms - 1,
            probs=Histogram.renormalize(probs),
        )

    @staticmethod
    def renormalize(probs: np.ndarray) -> np.ndarray:
        """Rectifies the inputted probabilities and renormalizes to unit sum,
        counteracting small numerical errors.

        Args:
            probs: A numpy.ndarray containing the probabilities to renormalize.

        Returns:
            numpy.ndarray containing the renormalized probabilities.

        Raises:
            ValueError: If the rectified probabilities sum to zero.
        """
        probs = np.maximum(0.0, probs)
        if np.allclose(np.sum(probs), 0.0):
            raise ValueError("Rectified probabilities sum to zero.")
        probs /= np.sum(probs, axis=-1)
        return probs

    @staticmethod
    def _mix(hists: List["Histogram"], weights: List[float]) -> "Histogram":
        if not isinstance(hists, list) or not isinstance(hists[0], Histogram):
            raise TypeError("input 'hists' must be a list of Histograms.")
        if not isinstance(weights, list) or not isinstance(weights[0], float):
            raise TypeError("input 'weights' must be a list of floats.")
        if len(hists) != len(weights):
            raise ValueError("inputs 'hists' and 'weight' must have same length.")
        if not np.allclose(sum(weights), 1.0):
            raise ValueError("input 'weights' must sum to one.")
        if min(weights) < 0.0:
            raise ValueError("input 'weights' must be all non-negative.")
        if len(set(h.vmin for h in hists)) != 1:
            raise ValueError("input 'hists' must have matching vmin.")
        if len(set(h.vmax for h in hists)) != 1:
            raise ValueError("input 'hists' must have matching vmax.")
        if len(set(h.num_atoms for h in hists)) != 1:
            raise ValueError("input 'hists' must have matching num_atoms.")

        new_probs = sum(weights[i] * hists[i].probs for i in range(len(hists)))
        return Histogram(
            vmin=hists[0].vmin,
            vmax=hists[0].vmax,
            num_atoms=hists[0].num_atoms,
            probs=Histogram.renormalize(new_probs),
        )
