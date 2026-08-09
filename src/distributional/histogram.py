import numpy as np
import matplotlib.pyplot as plt
from typing import Union


class Histogram:
    def __init__(self, first_atom: float, num_atoms: int, atom_stride: int, probs: np.ndarray) -> None:
        self.first_atom = first_atom
        self.num_atoms = num_atoms
        self.atom_stride = atom_stride
        self.probs = probs

    def __mul__(self: 'Histogram', coef: float) -> 'Histogram':
        return Histogram(
            first_atom=coef * self.first_atom,
            num_atoms=self.num_atoms,
            atom_stride=coef * self.atom_stride,
            probs=np.copy(self.probs),
        )

    def shift(self: 'Histogram', shift: float) -> 'Histogram':
        return Histogram(
            first_atom=shift + self.first_atom,
            num_atoms=self.num_atoms,
            atom_stride=self.atom_stride,
            probs=np.copy(self.probs),
        )

    def __add__(self: 'Histogram', other: Union['Histogram', float, int]) -> 'Histogram':
        if isinstance(other, int):
            return self.shift(other)
        if isinstance(other, float):
            return self.shift(other)
        assert isinstance(other, Histogram)

        assert self.first_atom == other.first_atom
        assert self.num_atoms == other.num_atoms
        assert self.atom_stride == other.atom_stride
        probs = np.zeros(dtype=self.probs.dtype, shape=[2 * self.num_atoms - 1])
        for i in range(0, len(self.atoms)):
            for j in range(0, len(self.atoms)):
                probs[i+j] += self.probs[i] * other.probs[j]
        return Histogram(
            first_atom=self.first_atom,
            num_atoms=2 * self.num_atoms - 1,
            atom_stride=self.atom_stride,
            probs=probs,
        )

    def convolve(self: 'Histogram', other: 'Histogram') -> 'Histogram':
        assert self.first_atom == other.first_atom
        assert self.num_atoms == other.num_atoms
        assert self.atom_stride == other.atom_stride
        # 1. Calculate the size of the output sequence
        conv_size = 2 * self.num_atoms - 1

        # 2. Compute RFFT for both signals, padded to the output size
        X = np.fft.rfft(self.probs, n=conv_size)
        Y = np.fft.rfft(other.probs, n=conv_size)

        # 3. Multiply in the frequency domain and invert back
        probs = np.fft.irfft(X * Y, n=conv_size)
        return Histogram(
            first_atom=self.first_atom,
            num_atoms=2 * self.num_atoms - 1,
            atom_stride=self.atom_stride,
            probs=probs,
        )

    @property
    def atoms(self):
        return np.arange(self.num_atoms) * self.atom_stride + self.first_atom

    def plot(self):
        bin_edges = np.concatenate([self.atoms - self.atom_stride / 2, np.array([self.atoms[-1] + self.atom_stride / 2])], axis=0)
        #widths = [bin_edges[i+1] - bin_edges[i] for i in range(len(bin_edges)-1)]
        #bin_centers = [bin_edges[i] + widths[i]/2 for i in range(len(bin_edges)-1)]
        widths = [self.atom_stride for _ in range(self.num_atoms)]
        bin_centers = self.atoms
        plt.bar(
            bin_centers,
            self.probs,
            width=widths,
            edgecolor="black",
            align="center"
        )
        plt.show()

    def rebin(self, num_atoms: int, new_vmin: float, new_vmax: float) -> 'Histogram':
        #def rebin(self, first_atom: float, num_atoms: int, atom_stride: float) -> 'Histogram':
        old_atoms = self.atoms
        old_vmin = old_atoms[0] - self.atom_stride / 2
        old_vmax = old_atoms[-1] + self.atom_stride / 2
        atom_stride = (new_vmax - new_vmin) / num_atoms
        print(f"atom_stride: {atom_stride}")
        assert (new_vmin <= old_vmin), f"missing left bin coverage of old distribution: (new_vmin > old_vmin), ({new_vmin} > {old_vmin})"
        assert (old_vmax <= new_vmax), f"missing right bin coverage of old distribution: (old_vmax > new_vmax), ({old_vmax} > {new_vmax})"

        old_left_atom_padding = []
        while (new_vmin < old_vmin):
            old_vmin -= self.atom_stride
            old_left_atom_padding.insert(0, old_vmin)
        old_left_atom_padding = np.array(old_left_atom_padding)

        old_right_atom_padding = []
        while (old_vmax < new_vmax):
            old_vmax += self.atom_stride
            old_right_atom_padding.append(old_vmax)
        old_right_atom_padding = np.array(old_left_atom_padding)

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

        new_atoms = np.arange(num_atoms) * atom_stride + new_vmin + atom_stride/2
        new_probs = np.zeros(dtype=self.probs.dtype, shape=[num_atoms])
        j = 0
        for i in range(0, num_atoms):
            # print(f"i,j: {i}, {j}")
            old_left = old_atoms[j] - self.atom_stride / 2
            old_right = old_atoms[j] + self.atom_stride / 2
            new_left = new_atoms[i] - atom_stride / 2
            new_right = new_atoms[i] + atom_stride / 2
            # print(f"old_left, old_right: {old_left}, {old_right}")
            # print(f"new_left, new_right: {new_left}, {new_right}")
            assert old_left <= new_left <= new_right, f"got (old_left, new_left, new_right) == ({old_left}, {new_left}, {new_right})"

            if new_right < old_right:
                frac = (new_right - new_left) / self.atom_stride
                new_probs[i] += frac * old_probs[j]
            if (new_right >= old_right):
                frac = (old_right - new_left) / self.atom_stride
                new_probs[i] += frac * old_probs[j]
                frac = (new_right - old_right) / self.atom_stride
                new_probs[i] += frac * (old_probs[j+1] if j+1 < old_probs.shape[0] else 0.0)
                j += 1
                #old_left = old_atoms[j] - self.atom_stride / 2
                #old_right = old_atoms[j] + self.atom_stride / 2

        return Histogram(
            first_atom=new_vmin + atom_stride/2,
            num_atoms=num_atoms,
            atom_stride=atom_stride,
            probs=new_probs,
        )


if __name__ == "__main__":
    hist = Histogram(first_atom=0, num_atoms=10, atom_stride=1, probs=np.array([0.1 for _ in range(10)]))
    hist.plot()

    hist2 = hist * 2.0
    hist2.plot()

    hist3 = hist * 0.5
    hist3.plot()

    hist4 = hist3 + 1.0
    hist4.plot()

    hist5 = hist + hist
    hist5.plot()

    hist6 = hist.convolve(hist)
    hist6.plot()

    print(hist5.probs)
    print(hist6.probs)

    hist10 = hist.rebin(new_vmin=-0.5, new_vmax=9.5, num_atoms=20)
    hist10.plot()
    print(np.sum(hist10.probs))

    hist11 = hist.rebin(new_vmin=-0.5, new_vmax=9.5, num_atoms=100)
    hist11.plot()
    print(np.sum(hist11.probs))

    # ((hist5 + -10.0) * 0.3).rebin(1000, -100, 100) + ((hist5 + 10.0) * 0.7).rebin(1000, -100, 100)

    # ((hist5 + -10.0) * 0.3).rebin(1000, -100, 100)

    # ((hist5 + -10.0) * 0.3).rebin(100, -20, 20).probs
