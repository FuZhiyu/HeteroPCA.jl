module HeteroPCA

using LinearAlgebra, Statistics, Random
import Base: size, show
using StatsBase
using StatsBase: CoefTable
import Statistics: mean, var
import StatsAPI: fit, predict, r2
import LinearAlgebra: eigvals, eigvecs

export HeteroPCAModel, fit, predict, reconstruct,
    projection, principalvars, r2, loadings, noisevars, var, eigvals, eigvecs,
    tprincipalvar, tresidualvar, heteropca, principalratio,
    StandardHeteroPCA, DeflatedHeteroPCA, DiagonalDeletion

# ────────────────────────────────────────────────────────────────────────────────
# A thin wrapper that mirrors the PCA struct defined in pca.jl
# ────────────────────────────────────────────────────────────────────────────────

"""
    HeteroPCAModel

Variant of PCA that jointly estimates the low‑rank common component **and**
heteroscedastic idiosyncratic noise variance.  Missing cells are allowed;
they are mean‑centered and then imputed with zero internally (i.e. the
"zero‑fill after demeaning" strategy proposed by Cai et al., 2019).

`proj` gives an **orthonormal** basis of the estimated factor space,
`prinvars` are the k largest singular values of the estimated low‑rank matrix,
and `noisevars` stores the final diagonal (noise) estimates.
"""
struct HeteroPCAModel{T<:Real}
    mean::Vector{T}      # d‑vector of variable means
    proj::Matrix{T}      # d×k orthonormal basis
    prinvars::Vector{T}      # length‑k singular values (≈ variance explained)
    tprinvar::T              # sum(prinvars)
    tvar::T              # total variance in data
    noisevars::Vector{T}      # length‑d heteroscedastic noise variances
    converged::Bool
    iterations::Int
end

# Short, PCA‑compatible helpers --------------------------------------------------

r2(M::HeteroPCAModel) = M.tprinvar / M.tvar
loadings(M::HeteroPCAModel) = sqrt.(principalvars(M))' .* projection(M)
noisevars(M::HeteroPCAModel) = M.noisevars

fullmean(d::Int, mv::AbstractVector{T}) where T = (isempty(mv) ? zeros(T, d) : mv)

## properties
"""
    size(M)

Returns a tuple with the dimensions of input (the dimension of the observation space)
and output (the dimension of the principal subspace).
"""
size(M::HeteroPCAModel) = size(M.proj)


"""
mean(M::HeteroPCAModel)

Returns the mean vector (of length `d`).
"""
mean(M::HeteroPCAModel) = fullmean(size(M.proj, 1), M.mean)

"""
    projection(M::HeteroPCAModel)

Returns the projection matrix (of size `(d, p)`). Each column of the projection matrix corresponds to a principal component.
The principal components are arranged in descending order of the corresponding variances.
"""
projection(M::HeteroPCAModel) = M.proj

"""
    eigvecs(M::HeteroPCAModel)

Get the eigenvalues of the PCA model `M`.
"""
eigvecs(M::HeteroPCAModel) = projection(M)

"""
    principalvars(M::HeteroPCAModel)

Returns the variances of principal components.
"""
principalvars(M::HeteroPCAModel) = M.prinvars
principalvar(M::HeteroPCAModel, i::Int) = M.prinvars[i]

"""
    eigvals(M::HeteroPCAModel)

Get the eigenvalues of the PCA model `M`.
"""
eigvals(M::HeteroPCAModel) = principalvars(M)

"""
    tprincipalvar(M::HeteroPCAModel)

Returns the total variance of principal components, which is equal to `sum(principalvars(M))`.
"""
tprincipalvar(M::HeteroPCAModel) = M.tprinvar

"""
    tresidualvar(M::HeteroPCAModel)

Returns the total residual variance.
"""
tresidualvar(M::HeteroPCAModel) = M.tvar - M.tprinvar

"""
    var(M::HeteroPCAModel)

Returns the total observation variance, which is equal to `tprincipalvar(M) + tresidualvar(M)`.
"""
var(M::HeteroPCAModel) = M.tvar

"""
    r2(M::HeteroPCAModel)
    principalratio(M::HeteroPCAModel)

Returns the ratio of variance preserved in the principal subspace, which is equal to `tprincipalvar(M) / var(M)`.
"""
const principalratio = r2



# ────────────────────────────────────────────────────────────────────────────────
# Algorithm dispatch types
# ────────────────────────────────────────────────────────────────────────────────

"""
    HeteroPCAAlgorithm

Abstract type for HeteroPCA algorithm implementations.
"""
abstract type HeteroPCAAlgorithm end

"""
    StandardHeteroPCA <: HeteroPCAAlgorithm

Standard iterative HeteroPCA algorithm using SVD and Robbins-Monro updates.
This is the original algorithm from Cai, Ma & Wu (2019).
"""
struct StandardHeteroPCA <: HeteroPCAAlgorithm end

"""
    DeflatedHeteroPCA <: HeteroPCAAlgorithm

Deflated HeteroPCA algorithm with adaptive block sizing for improved convergence.
Implements the deflation strategy with two main parameters:

# Fields
- `t_block::Int`: Number of iterations per deflation block (default: 10)
- `condition_number_threshold::Float64`: Threshold for determining well-conditioned blocks (default: 4.0)

# Constructor
    DeflatedHeteroPCA(; t_block::Int=10, condition_number_threshold::Real=4.0)
"""
struct DeflatedHeteroPCA <: HeteroPCAAlgorithm
    t_block::Int
    condition_number_threshold::Float64
    
    DeflatedHeteroPCA(; t_block::Int=10, condition_number_threshold::Real=4.0) = 
        new(t_block, condition_number_threshold)
end

"""
    DiagonalDeletion <: HeteroPCAAlgorithm

Diagonal-deletion PCA algorithm that performs a single SVD step on the 
off-diagonal covariance matrix without iteration. This provides a baseline
comparison method.
"""
struct DiagonalDeletion <: HeteroPCAAlgorithm end

# ────────────────────────────────────────────────────────────────────────────────
# Shared utility functions
# ────────────────────────────────────────────────────────────────────────────────

"""
    heteropca_step!(M, U, S, rank, α, check_convergence, abstol)

Unified HeteroPCA step: SVD, convergence check, and diagonal update in one function.
Returns (converged, error) where converged indicates if tolerance was met.
"""
function heteropca_step!(M::AbstractMatrix{T}, U::AbstractMatrix{T}, S::AbstractVector{T},
    rank::Int, α::Real, check_convergence::Bool, abstol::Real) where T<:Real

    # SVD step
    F = svd(M; full=false)
    U_k = @view F.U[:, 1:rank]
    S_k = @view F.S[1:rank]

    # Store results in pre-allocated arrays
    copyto!(@view(U[:, 1:rank]), U_k)
    copyto!(@view(S[1:rank]), S_k)

    # Combined convergence check and diagonal update in single pass
    converged = true
    err = zero(T)
    d = size(M, 1)

    @inbounds for i in 1:d
        # Compute new diagonal value: sum_j (U_k[i,j] * S_k[j] * U_k[i,j])
        new_diag = zero(T)
        for j in 1:rank
            new_diag += U_k[i, j] * S_k[j] * U_k[i, j]
        end

        # Check convergence if requested
        if check_convergence
            diff = abs(new_diag - M[i, i])
            err = max(err, diff)
            converged = converged && (diff < abstol)
        end

        # Update diagonal with relaxation
        M[i, i] = α * new_diag + (1 - α) * M[i, i]
    end

    return converged, err
end

"""
    heteropca_finalize(Σ, U, S, μ, rank, converged, iter)

Unified finalization step for all HeteroPCA algorithms.
Computes noise variances and creates the final HeteroPCAModel.
"""
function heteropca_finalize(Σ::AbstractMatrix{T}, U::AbstractMatrix{T}, S::AbstractVector{T},
    μ::AbstractVector{T}, rank::Int, converged::Bool, iter::Int) where T<:Real
    d = size(Σ, 1)

    # Compute reconstructed diagonal for noise variance estimation
    M̃_diag = Vector{T}(undef, d)
    @inbounds for i in 1:d
        M̃_diag[i] = zero(T)
        for j in 1:rank
            M̃_diag[i] += U[i, j] * S[j] * U[i, j]
        end
    end

    proj = U
    prinvars = S
    tprinvar = sum(S)
    noisevars = diag(Σ) .- M̃_diag
    tvar = tprinvar + sum(noisevars)

    return HeteroPCAModel(μ, proj, prinvars, tprinvar, tvar, noisevars, converged, iter)
end

# ────────────────────────────────────────────────────────────────────────────────
# Core algorithm (Cai, Ma & Wu 2019; Xue & Zou 2023)
# ────────────────────────────────────────────────────────────────────────────────

"""
    fit(HeteroPCAModel, X; rank=size(X, 1), kwargs...)

Estimate a HeteroPCAModel model from a `d × n` matrix `X` that **may contain
`missing`**.  

# Arguments
- `X::AbstractMatrix{T}`: The input data matrix of dimension `d × n`
- `rank::Int=size(X, 1)`: The target dimensionality *k* (number of components to extract)

# Keyword arguments
- `algorithm::HeteroPCAAlgorithm=StandardHeteroPCA()`: Algorithm to use (StandardHeteroPCA(), DeflatedHeteroPCA(), or DiagonalDeletion())
- `maxiter::Int=1_000`: Maximum number of iterations for the algorithm 
- `abstol::Float64=1e-6`: Convergence tolerance for diagonal estimation
- `demean::Bool=true`: Whether to center the data by subtracting column means; if the model is already demeaned, set `demean=false`
- `impute_method::Symbol=:pairwise`: Method for handling missing values (:pairwise or :zero); `impute = :pairwise` compute the pairwise covariance matrix using available data only; `impute = :zero` fills the missing values with zeros after demeaning, and compute the covariance matrix adjusted for the sample missing rate. 
- `α::Float64=1.0`: Diagonal update relaxation parameter; `α = 1` reproduces the original scheme;
- `suppress_warnings::Bool=false`: Whether to suppress convergence warnings

"""

"""
    _fit_impl(alg::StandardHeteroPCA, Σ, μ, rank; maxiter, abstol, α, suppress_warnings)

Implementation for standard HeteroPCA algorithm. Operates on the covariance matrix Σ.
"""
function _fit_impl(alg::StandardHeteroPCA, Σ::AbstractMatrix{T}, μ::AbstractVector{T}, rank::Int;
    maxiter::Int=1_000,
    abstol::Real=1e-6,
    α::Real=1.0,
    suppress_warnings::Bool=false) where T<:Real
    
    d = size(Σ, 1)

    # Pre-allocate work arrays
    M = Σ - Diagonal(diag(Σ))        # off‑diag part (allocates once)
    U = Matrix{T}(undef, d, rank)    # projection matrix storage
    S = Vector{T}(undef, rank)       # singular values storage

    iter = 0
    converged = false
    
    while iter < maxiter && !converged
        # Unified step: SVD, convergence check, and diagonal update
        converged, err = heteropca_step!(M, U, S, rank, α, true, abstol)
        iter += 1
    end

    iter == maxiter && !suppress_warnings && @warn "HeteroPCAModel: not converged after $maxiter iterations"

    return heteropca_finalize(Σ, U, S, μ, rank, converged, iter)
end

"""
    _fit_impl(alg::DeflatedHeteroPCA, Σ, μ, rank; maxiter, abstol, α, suppress_warnings)

Implementation for deflated HeteroPCA algorithm. Operates on the covariance matrix Σ.
"""
function _fit_impl(alg::DeflatedHeteroPCA, Σ::AbstractMatrix{T}, μ::AbstractVector{T}, rank::Int;
    maxiter::Int=1_000,
    abstol::Real=1e-6,
    α::Real=1.0,
    suppress_warnings::Bool=false) where T<:Real
    
    d = size(Σ, 1)
    M = Σ - Diagonal(diag(Σ))        # off‑diag part, consistent with standard algorithm

    # Run deflated algorithm on M
    U, M_final = deflated_heteropca_core(M, rank;
        t_block=alg.t_block,
        condition_number_threshold=alg.condition_number_threshold,
        α=α)
    
    # Compute principal variances from the converged M matrix
    # Project M onto the estimated subspace to get the variance explained
    M_projected = U' * M_final * U
    S = svdvals(Symmetric(M_projected))

    converged = true  # Deflated algorithm always "converges"
    iter = alg.t_block  # Report the block iterations used

    return heteropca_finalize(Σ, U, S, μ, rank, converged, iter)
end

"""
    _fit_impl(alg::DiagonalDeletion, Σ, μ, rank; maxiter, abstol, α, suppress_warnings)

Implementation for diagonal-deletion PCA algorithm. This is just one iteration 
of the standard algorithm without diagonal updates (α=0).
"""
function _fit_impl(alg::DiagonalDeletion, Σ::AbstractMatrix{T}, μ::AbstractVector{T}, rank::Int;
    maxiter::Int=1_000,
    abstol::Real=1e-6,
    α::Real=1.0,
    suppress_warnings::Bool=false) where T<:Real
    
    d = size(Σ, 1)

    # Pre-allocate work arrays  
    M = Σ - Diagonal(diag(Σ))        # off‑diag part
    U = Matrix{T}(undef, d, rank)    # projection matrix storage
    S = Vector{T}(undef, rank)       # singular values storage
    
    # Single step with α=0 (no diagonal update, just SVD of off-diagonal matrix)
    converged, err = heteropca_step!(M, U, S, rank, 0.0, false, 0.0)
    
    # Diagonal deletion always "converges" since it's a single operation
    converged = true
    iter = 1

    return heteropca_finalize(Σ, U, S, μ, rank, converged, iter)
end

function fit(::Type{HeteroPCAModel}, X::AbstractMatrix{T}, rank=size(X, 1);
    algorithm::HeteroPCAAlgorithm=StandardHeteroPCA(),
    maxiter=1_000,
    abstol=1e-6,
    demean=true,
    impute_method=:pairwise,
    α=1.0,
    suppress_warnings=false) where T

    d, n = size(X)
    
    # Data preprocessing (common to both algorithms)
    if demean
        μ = [mean(skipmissing(view(X, i, :))) for i in 1:d]
        Z = X .- μ
    else
        Z = X
        μ = fill(zero(eltype(X)), d)
    end

    if impute_method == :pairwise
        mimask = .!ismissing.(Z)
        Zfilled = coalesce.(Z, zero(eltype(Z)))
        Σ = (Zfilled * Zfilled') ./ (mimask * mimask' .- 1)
    elseif impute_method == :zero
        p = mean(!ismissing, Z)
        Zfilled = coalesce.(Z, zero(eltype(Z)))
        Σ = Zfilled * Zfilled' ./ (n - 1) ./ p^2
        for i in 1:d # diagonal is only off by p
            Σ[i, i] *= p
        end
    else
        @error "impute_method must be :pairwise or :zero. :pairwise is used."
        Σ = pairwise(cov, eachrow(Z), skipmissing=:pairwise)
    end

    return _fit_impl(algorithm, Σ, μ, rank; 
                    maxiter=maxiter, abstol=abstol, α=α, suppress_warnings=suppress_warnings)
end

# Predict / reconstruct (parallel to pca.jl) ------------------------------------

# --------------------------------------------------------------------------
# Mask‑aware predictions (column‑wise GLS with missing values) -------------
# --------------------------------------------------------------------------

"""
    predict(M::HeteroPCAModel, x::AbstractVector; λ = 0.0)

Project a **single observation vector** `x` onto the estimated factor space,
handling missing entries by solving a weighted least‑squares problem on the
observed coordinates only.

`λ` adds a small ridge (Tikhonov) regularisation to keep the system well‑posed
when fewer than `rank` coordinates are observed.
"""
function predict(M::HeteroPCAModel, x::AbstractVector{T}; λ::Real=0.0) where {T}
    obs = map(!ismissing, x)
    U = projection(M)
    k = size(U, 2)

    if count(obs) == 0
        return fill(zero(T), k)               # no information at all
    end

    Uobs = @view U[obs, :]
    μobs = @view M.mean[obs]
    zobs = coalesce.(x[obs] - μobs, zero(T))

    G = Uobs' * Uobs
    if λ != 0.0
        G .+= λ * I(k)
    end
    return G \ (Uobs' * zobs)
end


"""
    predict(M::HeteroPCAModel, X::AbstractMatrix; λ = 0.0)

Column‑wise version: returns a `k × n` matrix whose *j*‑th column is
`predict(M, X[:, j]; λ = λ)`.
"""
function predict(M::HeteroPCAModel, X::AbstractMatrix{T}; λ::Real=0.0) where {T}
    k, n = size(projection(M), 2), size(X, 2)
    Z = Matrix{T}(undef, k, n)
    @inbounds for j in 1:n
        Z[:, j] = predict(M, @view(X[:, j]); λ=λ)
    end
    return Z
end

"""
    reconstruct(M::HeteroPCAModel, y)

Inverse transform from factor space back to the original *d*‑space.
"""
function reconstruct(M::HeteroPCAModel, y::AbstractVecOrMat{T}) where {T}
    return M.proj * y .+ M.mean
end



function show(io::IO, M::HeteroPCAModel)
    idim, odim = size(M)
    print(io, "HeteroPCAModel(indim = $idim, outdim = $odim, principalratio = $(r2(M)))")
end

function show(io::IO, ::MIME"text/plain", M::HeteroPCAModel)
    idim, odim = size(M)
    print(io, "HeteroPCAModel(indim = $idim, outdim = $odim, principalratio = $(r2(M)))")
    ldgs = loadings(M)
    rot = diag(ldgs' * ldgs)
    ldgs = ldgs[:, sortperm(rot, rev=true)]
    ldgs_signs = sign.(sum(ldgs, dims=1))
    replace!(ldgs_signs, 0 => 1)
    ldgs = ldgs * diagm(0 => ldgs_signs[:])
    print(io, "\n\nPattern matrix (unstandardized loadings):\n")
    cft = CoefTable(ldgs, string.("PC", 1:odim), string.("", 1:idim))
    print(io, cft)
    print(io, "\n\n")
    print(io, "Importance of components:\n")
    λ = eigvals(M)
    prp = λ ./ var(M)
    prpv = λ ./ sum(λ)
    names = ["SS Loadings (Eigenvalues)",
        "Variance explained", "Cumulative variance",
        "Proportion explained", "Cumulative proportion"]
    cft = CoefTable(vcat(λ', prp', cumsum(prp)', prpv', cumsum(prpv)'),
        string.("PC", 1:odim), names)
    print(io, cft)
end

"""
    heteropca(X, rank=size(X, 1); kwargs...)

Convenience wrapper for `fit(HeteroPCAModel, X, rank=size(X, 1); kwargs...)`.

# Arguments
- `X::AbstractMatrix{T}`: The input data matrix of dimension `d × n`
- `rank::Int=size(X, 1)`: The target dimensionality *k* (number of components to extract)

# Keyword arguments
- `algorithm::HeteroPCAAlgorithm=StandardHeteroPCA()`: Algorithm to use (StandardHeteroPCA(), DeflatedHeteroPCA(), or DiagonalDeletion())
- `maxiter::Int=1_000`: Maximum number of iterations for the algorithm 
- `abstol::Float64=1e-6`: Convergence tolerance for diagonal estimation
- `demean::Bool=true`: Whether to center the data by subtracting column means; if the model is already demeaned, set `demean=false`
- `impute_method::Symbol=:pairwise`: Method for handling missing values (:pairwise or :zero); `impute = :pairwise` compute the pairwise covariance matrix using available data only; `impute = :zero` fills the missing values with zeros after demeaning, and compute the covariance matrix adjusted for the sample missing rate. 
- `α::Float64=1.0`: Diagonal update relaxation parameter; `α = 1` reproduces the original scheme;
- `suppress_warnings::Bool=false`: Whether to suppress convergence warnings

"""
function heteropca(X::AbstractMatrix{T}, rank=size(X, 1);
    algorithm::HeteroPCAAlgorithm=StandardHeteroPCA(),
    maxiter=1_000,
    abstol=1e-6,
    demean=true,
    impute_method=:pairwise,
    α=1.0,
    suppress_warnings=false) where T

    return fit(HeteroPCAModel, X, rank;
        algorithm=algorithm,
        maxiter=maxiter,
        abstol=abstol,
        demean=demean,
        impute_method=impute_method,
        α=α,
        suppress_warnings=suppress_warnings)
end

# ────────────────────────────────────────────────────────────────────────────────
# Deflated HeteroPCA algorithm (helper functions)
# ────────────────────────────────────────────────────────────────────────────────


"""
    heteropca_inner!(M, U, S, r, t_max, α)

Inner routine for deflated HeteroPCA using the unified step function.
"""
function heteropca_inner!(M::AbstractMatrix{T}, U::AbstractMatrix{T}, S::AbstractVector{T},
    r::Int, t_max::Int, α::Real=1.0) where T<:Real

    for _ in 1:t_max
        # No convergence check for deflated algorithm
        heteropca_step!(M, U, S, r, α, false, 0.0)
    end

    return M
end

"""
    find_next_block_size(s, r_start, r_end, condition_number_threshold)

Find the next deflation block size based on condition number and spectral gap criteria.
"""
function find_next_block_size(s::AbstractVector{T}, r_start::Int, r_end::Int, 
                             condition_number_threshold::Real) where T<:Real
    r_select = r_start
    
    for i in r_start:min(r_end, length(s))
        # Condition number check
        reference_singular_value = (r_start == 1) ? s[1] : s[r_start]
        if reference_singular_value / s[i] > condition_number_threshold
            break
        # Spectral gap check: σᵢ > (r/(r-1)) * σᵢ₊₁ OR we've reached the target rank
        elseif i == r_end || (i < length(s) && s[i] > (r_end/(r_end-1)) * s[i+1])
            r_select = i
        end
    end
    
    return r_select
end

"""
    deflated_heteropca_core(M_in, r; t_block=10, condition_number_threshold=4.0, α=1.0)

Core deflated HeteroPCA algorithm that implements the deflation strategy.
Takes the off-diagonal matrix M as input and returns both the subspace U and converged M.
"""
function deflated_heteropca_core(M_in::AbstractMatrix{T}, r::Int;
    t_block::Int=10, 
    condition_number_threshold::Real=4.0,
    α::Real=1.0) where T<:Real

    d = size(M_in, 1)
    G = copy(M_in)  # Work with a copy to preserve input
    
    # Pre-allocate arrays for efficiency
    U_hat = Matrix{T}(undef, d, r)
    U_temp = similar(U_hat)  # Temporary storage for eigenvectors
    S_temp = Vector{T}(undef, r)
    
    r_select = 0
    
    # Deflation loop: process components in adaptive blocks
    while r_select < r
        # Find next block size
        s = svdvals(G)
        r_new = find_next_block_size(s, r_select + 1, r, condition_number_threshold)
        
        if r_new > r_select  # Only proceed if we found new components
            # Run iterations for current block
            G = heteropca_inner!(G, U_temp, S_temp, r_new, t_block, α)
            U_hat[:, 1:r_new] = U_temp[:, 1:r_new]
            r_select = r_new
        else
            # No progress possible, break to avoid infinite loop
            break
        end
    end
    
    return U_hat, G
end

end # module