import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import logging

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

        Returns:
            A new Histogram instance representing the distribution of the new variable.

        Raises:
            TypeError: If vmin is not an int or float.
            TypeError: If vmax is not an int or float.
            TypeError: If num_atoms is not an int. 
            TypeError: If probs is not a numpy.ndarray.
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
        if len(probs.shape) != 1 or probs.shape[0] != num_atoms:
            raise ValueError("input 'probs' must be of shape (num_atoms,).")
        if not np.allclose(probs, np.abs(probs)):
            raise ValueError("input 'probs' must be non-negative.")
        if not np.allclose(np.sum(probs), 1.0):
            raise ValueError("input 'probs' must sum to one.")
        
        self.vmin = vmin
        self.vmax = vmax
        self.num_atoms = num_atoms
        self.probs = probs

    @property
    def atom_stride(self) -> float:
        """float: the histogram bin width.
        """
        return (self.vmax - self.vmin) / self.num_atoms  # num atoms = num bins
    
    @property
    def atom_min(self) -> float:
        """float: the center of the leftmost histogram bin.
        """
        return self.vmin + self.atom_stride / 2
    
    @property
    def atom_max(self) -> float:
        """float: the center of the rightmost histogram bin.
        """
        return self.vmax - self.atom_stride / 2
    
    @property
    def atoms(self) -> np.ndarray:
        """numpy.ndarray: the centers for the histogram bins.
        """
        output = np.arange(self.num_atoms) * self.atom_stride + self.atom_min
        np.testing.assert_allclose(output[-1], self.atom_max)
        return output
    
    def _shift(self: 'Histogram', shift: Union[int, float]) -> 'Histogram':
        if not isinstance(shift, int) and not isinstance(shift, float):
            raise TypeError("input 'shift' must be int or float type.")
        return Histogram(
            vmin=shift + self.vmin,
            vmax=shift + self.vmax,
            num_atoms=self.num_atoms,
            probs=np.copy(self.probs),
        )

    def _convolve(self: 'Histogram', other: 'Histogram') -> 'Histogram':
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
            probs=probs,
        )

    def _convolve_slow(self: 'Histogram', other: 'Histogram') -> 'Histogram':
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
                probs[i+j] += self.probs[i] * other.probs[j]
        return Histogram(
            vmin=self.atom_min + other.atom_min - self.atom_stride / 2,
            vmax=self.atom_max + other.atom_max + self.atom_stride / 2,
            num_atoms=2 * self.num_atoms - 1,
            probs=probs,
        )

    def __add__(self: 'Histogram', other: Union['Histogram', float, int]) -> 'Histogram':
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

    def __mul__(self: 'Histogram', coef: Union[int, float]) -> 'Histogram':
        """Multiplies the current histogram's random variable by a scalar. 

        Args:
            coef: An int or float.

        Returns:
            A new Histogram instance representing the distribution of the new variable.

        Raises:
            TypeError: If other is not an int or float.
        """
        if not isinstance(coef, int) and not isinstance(coef, float):
            raise TypeError("input 'coef' must be int or float type.")
        return Histogram(
            vmin=min(coef * self.vmin, coef * self.vmax),
            vmax=max(coef * self.vmin, coef * self.vmax),
            num_atoms=self.num_atoms,
            probs=np.copy(self.probs),
        )

    def plot(self) -> None:
        """Plot the histogram using matplotlib.
        """
        plt.bar(
            self.atoms,
            self.probs,
            width=self.atom_stride,
            edgecolor="black",
            align="center"
        )
        plt.show()

    def rebin(self, new_vmin: float, new_vmax: float, new_num_atoms: int) -> 'Histogram':
        """Rebin the histogram so that the probability mass of each old bin 
        is shared according to the proportion of its intersection with each new bin.

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
        old_atoms = self.atoms
        old_vmin = self.vmin
        old_vmax = self.vmax
        new_atom_stride = (new_vmax - new_vmin) / new_num_atoms
        if not (new_vmin <= old_vmin):
            raise ValueError(f"missing left bin coverage of old distribution: (new_vmin > old_vmin), ({new_vmin} > {old_vmin})")
        if not (old_vmax <= new_vmax):
            raise ValueError(f"missing right bin coverage of old distribution: (old_vmax > new_vmax), ({old_vmax} > {new_vmax})")

        # pad old distribution with left atoms of zero probability mass
        # until old bins exceed left range of new ones
        old_left_atom_padding = []
        while (new_vmin < old_vmin):
            old_vmin -= self.atom_stride
            old_left_atom_padding.insert(0, old_vmin + self.atom_stride / 2)
        # extra one mandatory padding
        old_vmin -= self.atom_stride
        old_left_atom_padding.insert(0, old_vmin + self.atom_stride / 2)
        old_left_atom_padding = np.array(old_left_atom_padding)

        # pad old distribution with right atoms of zero probability mass
        # until old bins exceed right range of new ones
        old_right_atom_padding = []
        while (old_vmax < new_vmax):
            old_vmax += self.atom_stride
            old_right_atom_padding.append(old_vmax - self.atom_stride / 2)
        # extra one mandatory padding
        old_vmax += self.atom_stride
        old_right_atom_padding.append(old_vmax - self.atom_stride / 2)
        old_right_atom_padding = np.array(old_right_atom_padding)

        old_atoms = np.concatenate([
            old_left_atom_padding,
            old_atoms,
            old_right_atom_padding,
        ], axis=0)
        old_probs = np.concatenate([
            np.zeros_like(old_left_atom_padding),
            self.probs,
            np.zeros_like(old_right_atom_padding),
        ], axis=0)

        # loop over new bins, figure out how much old probability mass they intersect with
        new_atoms = np.arange(new_num_atoms) * new_atom_stride + new_vmin + new_atom_stride/2
        new_probs = np.zeros(dtype=self.probs.dtype, shape=[new_num_atoms])
        j = 0
        for i in range(0, new_num_atoms):
            logging.debug(f"new i: {i}")
            # for each new i, we should only get to it once its left side is past
            # the left side of of old bin j. 
            old_left = old_atoms[j] - self.atom_stride / 2
            old_right = old_atoms[j] + self.atom_stride / 2
            new_left = new_atoms[i] - new_atom_stride / 2
            new_right = new_atoms[i] + new_atom_stride / 2
            if not (old_left <= new_left):
                raise RuntimeError(f"got old_left > new_left: {old_left} > {new_left}")

            # now we loop over old bins j and add in the contrib to new bin i
            while (old_left < new_right):
                logging.debug(f"current i,j: {i},{j}")
                old_left = old_atoms[j] - self.atom_stride / 2
                old_right = old_atoms[j] + self.atom_stride / 2
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
            probs=new_probs,
        )
