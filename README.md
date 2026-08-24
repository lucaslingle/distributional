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

#### Motivation

The insight powering this library is that a histogram can be converted to and from a piecewise constant density, enabling operations such as rebinning and inverse_cdf calculation to be carried out precisely.

#### Construction

The essential class is the Histogram, which can be constructed several ways. One way is directly, based on a minimum and maximum range, a number of bins, and a probability mass specification:
```
unif = Histogram(vmin=-1, vmax=1, num_atoms=2, probs=np.array([0.5, 0.5]))
```

Another way is from data:
```
emp = Histogram.empirical(np.random.normal(size=[10000]))
```
which by default automatically determines the number of bins from the number of datapoints.

#### Arithmetic

Histograms can be added, subtracted, shifted, and scaled:
```
h = Histogram.empirical(np.random.normal(size=[10000]))
h2 = 1 - 0.5 * h
```
These arithmetic operations are lifted from those performed directly on the underlying random variables.

Continuing, let's write
```
h3 = h2 + h
```
Addition/subtraction of histograms is treated as addition/subtraction of independent random variables, and is carried out via convolution.

#### Rebinning

Histograms involved in arithmetic operations may not have the same bins.

The addition and subtraction operators automatically rebin the operands to enable seamless arithmetic on histograms; e.g., h3 was computed without manual rebinning.

Manual rebinning is also possible:
```
h4 = h3.rebin(-10, 10, 500)
```

#### Beyond

The library also supports many other operations, such as cdf, inverse_cdf, conditioning the random variable to fall in an open interval, plotting, and summary statistics such as expectation, variance, median, mode, and differential entropy.

Distributions can also be formed via mixtures. Rebinning, padding with and trimming with zero-mass bins, and renormalizing to minimize numerical error are also supported.

These operations can be chained together to support complex pipelines, e.g.:
- conditioning on a union of open intervals (via a mixture of conditioned histograms)
- computing tail measures like expected shortfall (condition on quantile and take expectation)
