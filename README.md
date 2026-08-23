# distributional

[![Tests](https://github.com/lucaslingle/distributional/actions/workflows/pytest.yml/badge.svg)](https://github.com/lucaslingle/distributional/actions/workflows/pytest.yml)

Histogram operations library.

### Background

For continuous random variables that admit a density, a convenient nonparametric approximation is a piecewise constant version of that density.

If one integrates over each interval where the density is piecewise constant and replaces each original locally-constant density value by the value of the integral, one obtains the humble histogram.

This identity enables a number of useful operations like rebinning and inverse cdf calculation to be guided through the lens of the original density view.

The purpose of this library is unify many of the operations one might wish to perform on random variables, and to model these operations with corresponding piecewise constant densities, or equivalently, with their anodyne histogram representations.

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

For unit testing and docs dependencies, replace ```.``` with ```'.[dev]'```, ```'.[docs]```, or ```'.[dev,docs]'```.

### Documentation
To read online, you can go to https://distributional.readthedocs.io.

To read locally, first install the docs dependencies as outlined above then run
```
mkdocs serve
```
in the project directory.
