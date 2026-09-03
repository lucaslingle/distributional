# distributional

[![Tests](https://github.com/lucaslingle/distributional/actions/workflows/pytest.yml/badge.svg)](https://github.com/lucaslingle/distributional/actions/workflows/pytest.yml)
[![RTD](https://app.readthedocs.org/projects/distributional/badge/?version=latest&style=flat)](https://distributional.readthedocs.io/)
[![PyPI](https://img.shields.io/pypi/v/distributional)](https://pypi.org/project/distributional/)

Histogram operations library.

### Getting started

To install from PyPI, run
```
pip install distributional
```
For unit testing and docs dependencies, replace ```distributional``` with ```'distributional[dev]'```, ```'distributional[docs]'```, or ```'distributional[dev,docs]'```.

To build from source, run
```
git clone https://github.com/lucaslingle/distributional
cd distributional
pip install .
```
For unit testing and docs dependencies, replace ```.``` with ```'.[dev]'```, ```'.[docs]'```, or ```'.[dev,docs]'```.

### Documentation
To read the documentation online, you can go to https://distributional.readthedocs.io.

To read the documentation locally, install the docs dependencies then run ```mkdocs serve``` in the project directory.

### Basic usage

#### Construction

The essential class is the Histogram, which can be constructed several ways.

The first way is directly, using a min and max value, a number of atoms, and a probability mass array:
```
import numpy as np
unif1 = Histogram(vmin=-1, vmax=1, num_atoms=2, probs=np.array([0.5, 0.5]))
```

You can instead specify the atoms (bin centers), or the bin edges themselves:
```
unif2 = Histogram.from_atoms(atoms=np.array([-0.5, 0.5]), probs=np.array([0.5, 0.5]))
unif3 = Histogram.from_bins(bins=np.array([-1, 0, 1]), probs=np.array([0.5, 0.5]))
```
Note that `unif1`, `unif2` and `unif3` all have the same bins and probabilities.

If you have a dataset you're modeling, you can fit a histogram as follows:
```
data = np.random.normal(size=[10000])
empirical = Histogram.from_data(data)
```
By default, `Histogram.from_data` determines the number of atoms from the dataset size.

#### Arithmetic

Histograms can be added, subtracted, shifted, and scaled:
```
h = Histogram.from_data(np.random.normal(size=[10000]))
h2 = 1 - 0.5 * h
```
These arithmetic operations are lifted from those performed directly on the underlying random variables.

Continuing, let's write
```
h3 = h2 + h
```
Addition/subtraction of histograms is treated as addition/subtraction of independent random variables, and is carried out via convolution.

#### Rebinning

By default, the addition/subtraction operators automatically rebin the operands, to enable seamless arithmetic on histograms with different bins; e.g., h3 was computed without manual rebinning.

The automatic rebinning strategy is configurable via the autorebin argument ('none', 'count', or 'stride', default 'stride') — see the [API reference](https://distributional.readthedocs.io) for details.

Manual rebinning is also possible at any time:
```
h4 = h3.rebin(-10, 10, 500)
```
The rebin method redistributes probability mass according to the intersection between old and new bins.

#### Beyond

The library also supports many other operations, such as cdf, inverse_cdf/quantile, conditioning the random variable to fall in an interval, plotting, and summary statistics such as expectation, variance, median, mode, and differential entropy.

Distributions can also be formed via mixtures. Rebinning, padding with and trimming away zero-mass bins, and renormalizing to minimize numerical error are also supported.

These operations can be chained together to support complex pipelines, e.g.:

- We can condition on a union of intervals via a mixture of conditioned histograms:
```
h5 = Histogram.from_data(np.random.normal(size=[10000]))
left_tail = h5.condition(right=h5.quantile(0.05))
right_tail = h5.condition(left=h5.quantile(0.95))
tails = Histogram.from_mixture(hists=[left_tail, right_tail], weights=[0.5, 0.5])
```

- We can compute a tail measure like the expected shortfall at 5%:
```
es_05 = h5.condition(right=h5.quantile(0.05)).expectation
```
