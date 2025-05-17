# HeteroPCA

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://fuzhiyu.github.io/HeteroPCA.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://fuzhiyu.github.io/HeteroPCA.jl/dev)
[![Build Status](https://github.com/fuzhiyu/HeteroPCA.jl/workflows/CI/badge.svg)](https://github.com/fuzhiyu/HeteroPCA.jl/actions)
[![Coverage](https://codecov.io/gh/fuzhiyu/HeteroPCA.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/fuzhiyu/HeteroPCA.jl)

## Overview

HeteroPCA.jl is a Julia package for performing heterogeneous principal component analysis - a variant of PCA that handles heterogeneous data by iteratively updating the diagonal elements of the covariance matrix.

The package provides a consistent interface similar to other dimensionality reduction techniques in Julia's statistical ecosystem.

## Installation

```julia
using Pkg
Pkg.add("HeteroPCA")
```

## Usage Example

```julia
using HeteroPCA
using Plots

# Load or create your data (rows are variables, columns are observations)
X = rand(10, 100)  # 10 variables, 100 observations

# Fit the HeteroPCA model, requesting 3 components
model = fit(HeteroPCAResult, X, maxoutdim=3)

# Display model information
println(model)

# Project data to lower-dimensional space
Y = predict(model, X)

# Reconstruct data from its principal components
X_reconstructed = reconstruct(model, Y)

# Access model properties
loadings_matrix = projection(model)
variances = principalvars(model)
variance_ratio = principalratio(model)

# Visualize the results
scatter(Y[1,:], Y[2,:], title="First two principal components")
```

## How it Works

Heterogeneous PCA is useful when your variables have different scales or characteristics. The algorithm works by:

1. Starting with the covariance matrix with zeros on the diagonal
2. Iteratively updating the diagonal elements by mixing the current diagonal with the diagonal of a truncated SVD reconstruction
3. Stopping when the relative changes in diagonal elements fall below a threshold

This approach can often provide better results than standard PCA for heterogeneous data sources.

## API Reference

Key functions and types include:

- `HeteroPCAResult`: The type representing a fitted HeteroPCA model
- `fit(HeteroPCAResult, X)`: Fit a HeteroPCA model to data
- `predict(model, X)`: Project data onto principal components
- `reconstruct(model, Y)`: Reconstruct data from principal components
- `projection(model)`: Get the projection/loadings matrix
- `principalvars(model)`: Get the variances of principal components
- `principalratio(model)`: Get the proportion of total variance explained
