# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HeteroPCA.jl is a Julia package implementing the HeteroPCA algorithm for heteroscedastic PCA with missing data support. The algorithm jointly estimates low-rank common components and heteroscedastic noise variance, handling missing values through two imputation strategies.

## Key Commands

### Testing
```bash
julia --project -e "using Pkg; Pkg.test()"
```

### Documentation
```bash
julia --project=docs docs/make.jl
```

### Package Development
```bash
julia --project -e "using Pkg; Pkg.instantiate()"  # Install dependencies
julia --project -e "using Pkg; Pkg.develop(PackageSpec(path=\".\"))"  # Develop mode
```

### Running Simulations
```bash
julia simulations/replicateYanChenFan2024.jl
julia simulations/replicateZhangCaiWu2021.jl
julia simulations/figure4a_replication.jl
```

## Code Architecture

### Core Components

- **HeteroPCAModel**: Main type representing fitted models with fields for projection matrix, principal variances, noise variances, and convergence status
- **Two algorithms**: 
  - Standard iterative algorithm (`fit_standard`) using SVD and Robbins-Monro updates
  - Deflated algorithm (`fit_deflated`) with adaptive block sizing for better convergence
- **Missing data handling**: Two imputation strategies (`:pairwise` and `:zero`) implemented at the covariance matrix level

### Main API Functions

- `heteropca(X, rank)` or `fit(HeteroPCAModel, X, rank)`: Fit model to data
- `predict(model, X)`: Project data onto principal components (handles missing values)
- `reconstruct(model, Y)`: Inverse transform from factor space
- Model inspection: `loadings()`, `projection()`, `principalvars()`, `noisevars()`, `r2()`

### File Structure

- `src/HeteroPCA.jl`: Complete implementation in single file (530 lines)
- `test/test_util.jl`: Utility function tests with synthetic data
- `test/test_simulation.jl`: Simulation tests with statistical validation
- `simulations/`: Replication scripts for published results
- `docs/`: Documenter.jl setup for documentation generation

### Algorithm Implementation Details

The core algorithm operates on covariance matrices rather than raw data:
1. Preprocessing: Handle missing values via pairwise covariance or zero-fill strategies
2. Standard algorithm: Iterative SVD with diagonal updates until convergence
3. Deflated algorithm: Block-wise deflation with adaptive sizing based on spectral properties
4. Both algorithms use the off-diagonal covariance matrix with iterative diagonal estimation

### Testing Strategy

- Comprehensive unit tests for all API functions
- Simulation tests with known ground truth
- Missing data scenarios with both imputation methods
- Statistical validation over multiple replications
- Tests run on Julia 1.3, 1.10, and nightly via GitHub Actions

### Missing Data Philosophy

The package handles missing values at the covariance matrix level, not through explicit imputation of individual data points. This approach:
- Maintains statistical validity
- Avoids introducing bias from imputation
- Supports two strategies: pairwise covariance (default) and zero-fill with adjustment