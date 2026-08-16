import logging
import math
from typing import Optional
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
    def expectation(self) -> float:
        """float: The expectation (mean) of the histogram."""
        return np.sum(self.atoms * self.probs, axis=-1)

    @property
    def variance(self) -> float:
        """float: The variance of the histogram."""
        mu = self.expectation
        return np.sum(np.square(self.atoms - mu) * self.probs, axis=-1)

    def _shift(self, shift: Union[int, float]) -> "Histogram":
        if not isinstance(shift, int) and not isinstance(shift, float):
            raise TypeError("input 'shift' must be int or float type.")
        return Histogram(
            vmin=shift + self.vmin,
            vmax=shift + self.vmax,
            num_atoms=self.num_atoms,
            probs=self.probs,  # probs is already a deep copy of self._probs
        )

    def _convolve(self, other: "Histogram") -> "Histogram":
        if not isinstance(other, Histogram):
            raise TypeError("input 'other' must be Histogram type.")
        if self.vmin != other.vmin:
            raise ValueError("other.vmin must equal self.vmin")
        if self.vmax != other.vmax:
            raise ValueError("other.vmax must equal self.vmax")
        if self.num_atoms != other.num_atoms:
            raise ValueError("other.num_atoms must equal self.num_atoms")

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

    def _convolve_slow(self, other: "Histogram") -> "Histogram":
        if not isinstance(other, Histogram):
            raise TypeError("input 'other' must be Histogram type.")
        if self.vmin != other.vmin:
            raise ValueError("other.vmin must equal self.vmin")
        if self.vmax != other.vmax:
            raise ValueError("other.vmax must equal self.vmax")
        if self.num_atoms != other.num_atoms:
            raise ValueError("other.num_atoms must equal self.num_atoms")

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
        if isinstance(other, int):
            return self._shift(float(other))
        if isinstance(other, float):
            return self._shift(other)
        if isinstance(other, Histogram):
            return self._convolve(other)
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
        if isinstance(other, int):
            return self._shift(-float(other))
        if isinstance(other, float):
            return self._shift(-other)
        if isinstance(other, Histogram):
            return self._convolve(-other)
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

    def pad(self, left: float, right: float, extra: bool = False) -> "Histogram":
        """Pad the histogram with zero-mass bins until the outer edges exceed the range given.

        Args:
            left: Left pad target.
            right: Right pad target.
            extra: Adds one extra atom to each side of the new histogram,
                beyond what is needed to cover the range specified. Defaults to False.

        Returns:
            New Histogram whose atoms minimally contain the range [left, right].

        Raises:
            ValueError: If left >= right.
            ValueError: If self.vmin < left.
            ValueError: If right < self.vmax.
        """
        if left >= right:
            raise ValueError("input 'left' must be less than 'right'.")
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

    def rebin(
        self, new_vmin: float, new_vmax: float, new_num_atoms: int
    ) -> "Histogram":
        """Rebin the histogram.

        Implemented so that the probability mass of each old bin is shared according
        to the proportion of its intersection with each new bin.

        Args:
            new_vmin: Minimum permitted value for the new histogram's random variable.
            new_vmax: Maximum permitted value for the new histogram's random variable.
            new_num_atoms: Number of bins for the new histogram.

        Returns:
            A new Histogram instance with the rebinned probability mass.

        Raises:
            ValueError: If the new_vmin is larger than the old one.
            ValueError: If the new_vmax is smaller than the old one.
            RuntimeError: If the algorithm does not function as expected.
                This should never occur.
        """
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
    def empirical(vs: np.ndarray, num_atoms: Optional[int] = None) -> "Histogram":
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
