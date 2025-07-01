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
    tprincipalvar, tresidualvar, heteropca, principalratio

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

abstract type HeteroPCAAlgorithm end

struct StandardHeteroPCA <: HeteroPCAAlgorithm end

struct DeflatedHeteroPCA <: HeteroPCAAlgorithm
    t_block::Int
    condition_number_threshold::Float64
    
    DeflatedHeteroPCA(; t_block::Int=10, condition_number_threshold::Real=4.0) = 
        new(t_block, condition_number_threshold)
end

struct DiagonalDeletion <: HeteroPCAAlgorithm end

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
- `algorithm::Symbol=:standard`: Algorithm to use (:standard, :deflated, or :diagonal_deletion)
- `maxiter::Int=1_000`: Maximum number of iterations for the algorithm 
- `abstol::Float64=1e-6`: Convergence tolerance for diagonal estimation
- `demean::Bool=true`: Whether to center the data by subtracting column means; if the model is already demeaned, set `demean=false`
- `impute_method::Symbol=:pairwise`: Method for handling missing values (:pairwise or :zero); `impute = :pairwise` compute the pairwise covariance matrix using available data only; `impute = :zero` fills the missing values with zeros after demeaning, and compute the covariance matrix adjusted for the sample missing rate. 
- `alpha::Float64=1.0`: Diagonal update relaxation parameter; `alpha = 1` reproduces the original scheme;
- `suppress_warnings::Bool=false`: Whether to suppress convergence warnings

# Deflated algorithm specific parameters
- `t_block::Int=10`: Iterations per deflation block (deflated algorithm only)
- `condition_number_threshold::Float64=4.0`: Well-conditioning threshold (deflated algorithm only)
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
    M = Σ - Diagonal(diag(Σ))        # off‑diag part
    U = Matrix{T}(undef, d, rank)    # preallocation
    S = Vector{T}(undef, rank)
    M̃ = similar(M)                   # pre-define M̃ outside the loop
    iter = 0
    converged = false
    
    while iter < maxiter
        F = svd(M; full=false)           # Truncated SVD (rank ≥ k)
        U = F.U[:, 1:rank]
        S = F.S[1:rank]
        M̃ = U * Diagonal(S) * U'        # best rank‑k approx (sym)
        err = maximum(abs.((diag(M̃) .- diag(M))))
        if err < abstol
            converged = true
            break
        end

        # Robbins–Monro diagonal update
        @inbounds for i in 1:d
            M[i, i] = α * M̃[i, i] + (1 - α) * M[i, i]
        end
        iter += 1
    end
    iter == maxiter && !suppress_warnings && @warn "HeteroPCAModel: not converged after $maxiter iterations"

    proj = U
    prinvars = S
    tprinvar = sum(S) 
    noisevars = diag(Σ) .- diag(M̃)
    tvar = tprinvar + sum(noisevars)

    return HeteroPCAModel(μ, proj, prinvars, tprinvar, tvar, noisevars, converged, iter)
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
    
    # Work directly with the covariance matrix Σ
    U = deflated_heteropca_core(Σ, rank; 
                              t_block=alg.t_block, 
                              condition_number_threshold=alg.condition_number_threshold)
    
    # Compute the reconstructed low-rank matrix
    M̃ = U * U'
    
    # Extract singular values from the reconstructed matrix  
    F = svd(M̃; full=false)
    S = F.S[1:rank]
    
    proj = U
    prinvars = S
    tprinvar = sum(S)
    noisevars = diag(Σ) .- diag(M̃)
    tvar = tprinvar + sum(noisevars)
    converged = true  # Deflated algorithm always "converges"
    iter = alg.t_block  # Report the block iterations used

    return HeteroPCAModel(μ, proj, prinvars, tprinvar, tvar, noisevars, converged, iter)
end

"""
    _fit_impl(alg::DiagonalDeletion, Σ, μ, rank; maxiter, abstol, α, suppress_warnings)

Implementation for diagonal-deletion PCA algorithm. This is essentially one-step SVD 
on the off-diagonal covariance matrix without iteration.
"""
function _fit_impl(alg::DiagonalDeletion, Σ::AbstractMatrix{T}, μ::AbstractVector{T}, rank::Int;
    maxiter::Int=1_000,
    abstol::Real=1e-6,
    α::Real=1.0,
    suppress_warnings::Bool=false) where T<:Real
    
    d = size(Σ, 1)
    M = Σ - Diagonal(diag(Σ))        # off‑diag part
    
    # Single SVD step (no iteration)
    F = svd(M; full=false)
    U = F.U[:, 1:rank]
    S = F.S[1:rank]
    M̃ = U * Diagonal(S) * U'        # best rank‑k approx (sym)
    
    proj = U
    prinvars = S
    tprinvar = sum(S) 
    noisevars = diag(Σ) .- diag(M̃)
    tvar = tprinvar + sum(noisevars)
    
    # Diagonal deletion always "converges" since it's a single operation
    converged = true
    iter = 1

    return HeteroPCAModel(μ, proj, prinvars, tprinvar, tvar, noisevars, converged, iter)
end

function fit(::Type{HeteroPCAModel}, X::AbstractMatrix{T}, rank=size(X, 1);
    algorithm::Symbol=:standard,
    maxiter=1_000,
    abstol=1e-6,
    demean=true,
    impute_method=:pairwise,
    α=1.0,
    suppress_warnings=false,
    t_block::Int=10,
    condition_number_threshold::Real=4.0) where T

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

    # Dispatch to appropriate algorithm
    if algorithm == :standard
        alg = StandardHeteroPCA()
    elseif algorithm == :deflated
        alg = DeflatedHeteroPCA(t_block=t_block, 
                               condition_number_threshold=condition_number_threshold)
    elseif algorithm == :diagonal_deletion
        alg = DiagonalDeletion()
    else
        throw(ArgumentError("algorithm must be :standard, :deflated, or :diagonal_deletion, got :$algorithm"))
    end
    
    return _fit_impl(alg, Σ, μ, rank; 
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
- `algorithm::Symbol=:standard`: Algorithm to use (:standard, :deflated, or :diagonal_deletion)
- `maxiter::Int=1_000`: Maximum number of iterations for the algorithm 
- `abstol::Float64=1e-6`: Convergence tolerance for diagonal estimation
- `demean::Bool=true`: Whether to center the data by subtracting column means; if the model is already demeaned, set `demean=false`
- `impute_method::Symbol=:pairwise`: Method for handling missing values (:pairwise or :zero); `impute = :pairwise` compute the pairwise covariance matrix using available data only; `impute = :zero` fills the missing values with zeros after demeaning, and compute the covariance matrix adjusted for the sample missing rate. 
- `alpha::Float64=1.0`: Diagonal update relaxation parameter; `alpha = 1` reproduces the original scheme;
- `suppress_warnings::Bool=false`: Whether to suppress convergence warnings

# Deflated algorithm specific parameters
- `t_block::Int=10`: Iterations per deflation block (deflated algorithm only)
- `condition_number_threshold::Float64=4.0`: Well-conditioning threshold (deflated algorithm only)
"""
function heteropca(X::AbstractMatrix{T}, rank=size(X, 1);
    algorithm::Symbol=:standard,
    maxiter=1_000,
    abstol=1e-6,
    demean=true,
    impute_method=:pairwise,
    α=1.0,
    suppress_warnings=false,
    t_block::Int=10,
    condition_number_threshold::Real=4.0) where T

    return fit(HeteroPCAModel, X, rank;
        algorithm=algorithm,
        maxiter=maxiter,
        abstol=abstol,
        demean=demean,
        impute_method=impute_method,
        α=α,
        suppress_warnings=suppress_warnings,
        t_block=t_block,
        condition_number_threshold=condition_number_threshold)
end

# ────────────────────────────────────────────────────────────────────────────────
# Deflated HeteroPCA algorithm (helper functions)
# ────────────────────────────────────────────────────────────────────────────────

"""
    Pdiag(A)

Keep only diagonal of square matrix `A`, zero elsewhere.
"""
function Pdiag(A::AbstractMatrix{T}) where T<:Real
    return Diagonal(diag(A))
end

"""
    Poff(A)

Zero the diagonal of `A`, keep the rest.
"""
function Poff(A::AbstractMatrix{T}) where T<:Real
    return A - Pdiag(A)
end

"""
    eigsr(A, r)

Return the column-orthonormal matrix (n×r) of top-r eigenvectors of symmetric `A`.
"""
function eigsr(A::AbstractMatrix{T}, r::Int) where T<:Real
    F = eigen(Symmetric(A))
    idx = sortperm(F.values, rev=true)
    return F.vectors[:, idx[1:r]]
end

"""
    heteropca_inner(G_in, r, t_max)

Inner routine for deflated HeteroPCA: iteratively impute better diagonal 
while keeping off-diagonal fixed.
"""
function heteropca_inner(G_in::AbstractMatrix{T}, r::Int, t_max::Int) where T<:Real
    G = copy(G_in)
    for _ in 1:t_max
        U = eigsr(G, r)
        G = Poff(G_in) + Pdiag(U * U')
    end
    return eigsr(G, r)
end

"""
    deflated_heteropca_core(Σ, r; t_block=10, condition_number_threshold=4.0, gap_threshold_factor=1.0)

Core deflated HeteroPCA algorithm that implements the deflation strategy following the R reference implementation.
Takes the covariance matrix Σ as input and implements the two-phase deflation process.
"""
function deflated_heteropca_core(Σ::AbstractMatrix{T}, r::Int; 
    t_block::Int=10, 
    condition_number_threshold::Real=4.0) where T<:Real
    
    d = size(Σ, 1)
    G = Poff(Σ)  # Start with off-diagonal covariance matrix
    
    # Phase 1: Initial block selection (following R implementation lines 2-20)
    r_select = 1
    
    # Find initial block size based on condition number and spectral gap
    s = svd(G, full=false).S
    for i in 1:r
        # Condition number check: σ₁ / σᵢ > condition_number_threshold
        if length(s) >= i && s[1] / s[i] > condition_number_threshold
            break
        # Spectral gap check: σᵢ > (r/(r-1)) * σᵢ₊₁ OR we've reached the target rank
        elseif i == r || (length(s) >= i+1 && s[i] > (r/(r-1)) * s[i+1])
            r_select = i
        end
    end
    
    # Run initial block iterations if r_select >= 1
    U_hat = Matrix{T}(undef, d, r)
    if r_select >= 1
        U1_hat = eigsr(G, r_select)
        
        # Iterative refinement for initial block
        for _ in 1:t_block
            # Update diagonal: G = G - diag(G) + diag(U₁ᵀU₁G)
            G = Poff(G) + Pdiag(U1_hat * U1_hat' * G)
            U1_hat = eigsr(G, r_select)
        end
        
        # Final update for this block
        G = Poff(G) + Pdiag(U1_hat * U1_hat' * G)
        U_hat[:, 1:r_select] = U1_hat
    end
    
    # Phase 2: Continue deflation for remaining components (following R implementation lines 21-35)
    while r - r_select >= 1
        r_temp = r_select
        
        # Find next block size
        s = svd(G, full=false).S
        for i in (r_temp+1):r
            # Condition number check: σᵣ_temp+1 / σᵢ > condition_number_threshold  
            if length(s) >= i && length(s) >= r_temp+1 && s[r_temp+1] / s[i] > condition_number_threshold
                break
            # Spectral gap check: σᵢ > (r/(r-1)) * σᵢ₊₁ OR we've reached the target rank
            elseif i == r || (length(s) >= i+1 && s[i] > (r/(r-1)) * s[i+1])
                r_select = i
            end
        end
        
        # Run iterations for current block
        for _ in 1:t_block
            U_current = eigsr(G, r_select)
            # Update diagonal: G = G - diag(G) + diag(UᵀUG)
            G = Poff(G) + Pdiag(U_current * U_current' * G)
        end
        
        # Store the current block result
        U_hat[:, 1:r_select] = eigsr(G, r_select)
    end
    
    return U_hat
end

end # module