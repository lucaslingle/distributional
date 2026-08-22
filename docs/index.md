# Distributional

Histogram operations library.

For continuous random variables that admit a density, a convenient nonparametric approximation is a piecewise constant version of that density.

If one integrates over each interval where the density is piecewise constant and replaces each original locally-constant density value by the value of the integral, one obtains the humble histogram.

This identity enables a number of useful operations like rebinning and inverse cdf calculation to be guided through the lens of the original density view.

The purpose of this library is unify many of the operations one might wish to perform on random variables, and to model these operations with corresponding piecewise constant densities, or equivalently, with their anodyne histogram representations.

# API Reference

::: distributional.histogram
    options:
      show_root_heading: true
      members_order: source
