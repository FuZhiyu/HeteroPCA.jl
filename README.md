# HeteroPCA.jl

[![Build Status](https://github.com/fuzhiyu/HeteroPCA.jl/workflows/CI/badge.svg)](https://github.com/fuzhiyu/HeteroPCA.jl/actions)

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://fuzhiyu.github.io/HeteroPCA.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://fuzhiyu.github.io/HeteroPCA.jl/dev) -->
<!-- [![Coverage](https://codecov.io/gh/fuzhiyu/HeteroPCA.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/fuzhiyu/HeteroPCA.jl) -->

## Overview

HeteroPCA.jl implements the HeteroPCA algorithm proposed by Zhang, Cai, and Wu (2021), along with the deflated HeteroPCA algorithm by Zhou and Chen (2025) for improved convergence. The API is similar to [MultivariateStats.jl](https://github.com/JuliaStats/MultivariateStats.jl) with extensions to handle missing values and multiple algorithm variants.  

## Installation

As it is currently unregistered, you can install it directly from this Github repo:
```julia
using Pkg
Pkg.add("https://github.com/FuZhiyu/HeteroPCA.jl")
```

## Basic Usage

```julia
using HeteroPCA
using LinearAlgebra, Random

# Create a simple factor model with heteroscedastic noise
d, n, r = 10, 500, 1  # 10 variables, 500 observations, 2 factors

# Generate true factor loadings (orthonormal)
U_true = qr(randn(d, r)).Q[:, 1]

# Generate factor scores
F_true = randn(r, n)

# Generate heteroscedastic noise with different variances per variable
noise_std = 0.1 .+ 0.4 * rand(d)  # Different noise levels for each variable
noise = noise_std .* randn(d, n)
# Create observed data matrix: X = U*F + noise
X = U_true * F_true + noise

# Fit the HeteroPCA model, requesting 2 components (the true rank)
model = heteropca(X, 1)
# Alternatively: model = fit(HeteroPCAModel, X, 2)

# access factor loadings
Uhat = loadings(model)
cor(vec(Uhat), vec(U_true)) # 0.99

# predict the factors based on observed data
Y = predict(model, X)
cor(vec(Y), vec(F_true)) # 0.96

# the reconstructed data (predicted X based on factors)
X_reconstructed = reconstruct(model, Y)
cor(vec(X_reconstructed), vec(U_true * F_true)) # 0.96
```

## Algorithm Options

HeteroPCA.jl supports three different algorithms:

- `StandardHeteroPCA()` (default): The original iterative HeteroPCA algorithm from Zhang, Cai & Wu (2021)
- `DeflatedHeteroPCA()`: Deflated HeteroPCA with adaptive block sizing for improved convergence
- `DiagonalDeletion()`: Simple diagonal-deletion PCA (baseline comparison)

```julia
# Standard algorithm (default)
model1 = heteropca(X, rank)

# Deflated algorithm with custom parameters
model2 = heteropca(X, rank; algorithm=DeflatedHeteroPCA(t_block=5, condition_number_threshold=10.0))

# Diagonal deletion (a wrapper with maxiter = 1 for standard algorithm)
model3 = heteropca(X, rank; algorithm=DiagonalDeletion())
```

## Keyword Arguments for `heteropca`

The `heteropca` function supports the following keyword arguments:

```julia
heteropca(X, rank = size(X, 1); 
    algorithm=StandardHeteroPCA(), 
    maxiter=1_000, 
    abstol=1e-6, 
    demean=true, 
    impute_method=:pairwise, 
    α=1.0,
    suppress_warnings=false)
```

### Parameters
- `algorithm::HeteroPCAAlgorithm=StandardHeteroPCA()`: Algorithm to use (StandardHeteroPCA(), DeflatedHeteroPCA(), or DiagonalDeletion())
- `maxiter::Int=1_000`: Maximum number of iterations for the algorithm
- `abstol::Float64=1e-6`: Convergence tolerance for diagonal estimation
- `demean::Bool=true`: Whether to center the data by subtracting column means; if the model is already demeaned, set `demean=false`
- `impute_method::Symbol=:pairwise`: Method for handling missing values (`:pairwise` or `:zero`)
- `α::Float64=1.0`: Diagonal update relaxation parameter; `α = 1` reproduces the original scheme
- `suppress_warnings::Bool=false`: Whether to suppress convergence warnings

### Algorithm-Specific Parameters

For `DeflatedHeteroPCA`, you can specify:
- `t_block::Int=10`: Number of iterations per deflation block
- `condition_number_threshold::Float64=4.0`: Threshold for determining well-conditioned blocks

```julia
# Example with custom deflated parameters
model = heteropca(X, rank; algorithm=DeflatedHeteroPCA(t_block=15, condition_number_threshold=6.0))
```

## Working with Missing Data

HeteroPCA naturally handles missing data:

```julia
# Create data with missing values

X_with_missing = convert(Matrix{Union{Float64,Missing}}, copy(X))
X_with_missing[rand(size(X)...).<0.5] .= missing
# Fit model (missing values are handled automatically)
model_missing = heteropca(X_with_missing, 1)

# predict the factors based on nonmissing data
Y_pred_missing = predict(model_missing, X_with_missing)
cor(vec(Y_pred_missing), vec(F_true)) # 0.92
```

There are two approaches to handle the missing values, controlled by `impute`. By default, `impute = :pairwise` compute the pairwise covariance matrix using available data only. The potential drawback of this approach is that the sample covariance matrix is not gauranteed to be positive-definite. In these cases, the alternative approach `impute = :zero` fills the missing values with zero after demeaning, and compute the covariance matrix adjusted for the sample missing rate. 


## API Reference

### Main Types and Functions


- `HeteroPCAModel`: The type representing a fitted HeteroPCA model
- `heteropca(X, rank)` or `fit(HeteroPCAModel, X, rank)`: Convenience function to fit a HeteroPCA model to data
- `predict(model, X)`: Project data onto principal components
- `reconstruct(model, Y)`: Reconstruct data from principal components

### Model Properties

- `loadings(model)`: Get the loading matrix
- `projection(model)`: Get the projection matrix (orthonormal basis)
- `principalvars(model)`: Get the variances of principal components
- `r2(model)` or `principalratio(model)`: Get the proportion of variance explained
- `noisevars(model)`: Get the estimated heteroscedastic noise variances
- `mean(model)`: Get the mean vector used for centering


## Replication of Published Results

Below is a replication of Figure 2 from Yan, Chen & Fan (2024), showing how HeteroPCA outperforms standard SVD-based methods for different sampling rates:

![Replication of Figure 2](https://github.com/FuZhiyu/HeteroPCA.jl/blob/e79b44a0884ec60b407b532b5749d7a94542d377/simulations/YanChenFan_figure2_replication.png)

To reproduce this figure, run the script `simulations/replicateYanChenFan2024.jl`.

See the folder `simulations` for more replications.

## References

- Zhang, A. R., Cai, T. T., & Wu, Y. (2021). Heteroskedastic PCA: Algorithm, Optimality, and Applications (No. arXiv:1810.08316). arXiv. https://doi.org/10.48550/arXiv.1810.08316
- Yuchen Zhou. Yuxin Chen. "Deflated HeteroPCA: Overcoming the curse of ill-conditioning in heteroskedastic PCA." Ann. Statist. 53 (1) 91 - 116, February 2025. https://doi.org/10.1214/24-AOS2456