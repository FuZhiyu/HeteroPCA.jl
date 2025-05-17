module HeteroPCA

using LinearAlgebra, Statistics, Random, Missings
import Base: size, show
import StatsModels: fit
using StatsBase: CoefTable
import Statistics: mean, var
import StatsAPI: fit, predict, r2
import LinearAlgebra: eigvals, eigvecs

export HeteroPCAModel, fit, predict, reconstruct,
    projection, principalvars, r2, loadings, diagnoise, var, eigvals, eigvecs,
    tprincipalvar, tresidualvar, heteropca

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
and `diagnoise` stores the final diagonal (noise) estimates.
"""
struct HeteroPCAModel{T<:Real}
    mean::Vector{T}      # d‑vector of variable means
    proj::Matrix{T}      # d×k orthonormal basis
    prinvars::Vector{T}      # length‑k singular values (≈ variance explained)
    tprinvar::T              # sum(prinvars)
    tvar::T              # total variance in data
    diagnoise::Vector{T}      # length‑d heteroscedastic noise variances
    converged::Bool
    iterations::Int
end

# Short, PCA‑compatible helpers --------------------------------------------------

principalvars(M::HeteroPCAModel) = M.prinvars
r2(M::HeteroPCAModel) = M.tprinvar / M.tvar
loadings(M::HeteroPCAModel) = sqrt.(principalvars(M))' .* projection(M)
diagnoise(M::HeteroPCAModel) = M.diagnoise

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
################################################################################
# Mean‑centring helpers that tolerate `missing`
################################################################################

"""
    centralize(X, mean)

Return a *numeric* copy of `X` (vector or `d×n` matrix) in which each row is
mean‑centred.  After centring, any `missing` entry is replaced by **0.0**,
recreating the same convention used when HeteroPCAModel was fitted.

The original `X` is left untouched; the result is `Float64`.
"""
function centralize(X::AbstractVector, mean::AbstractVector)
    isempty(mean) && return replace!(copy(X), missing => 0.0)

    # Vectorized operation with broadcasting
    return coalesce.(X .- mean, 0.0)
end

function centralize(X::AbstractMatrix, mean::AbstractVector)
    isempty(mean) && return replace!(copy(X), missing => 0.0)

    # Vectorized operation with broadcasting
    # Reshape mean as column vector for row-wise broadcasting
    return coalesce.(X .- mean, 0.0)
end

"""
    decentralize(Z, mean)

Add back the row means stored in `mean` to the **centred** numeric data `Z`.
"""
decentralize(Z::AbstractVecOrMat{T}, mean::AbstractVector) where {T<:Real} =
    isempty(mean) ? Z : Z .+ mean


"""
    demean(X)

Convenience wrapper that returns **both** the centred/imputed matrix and the
per‑row means:

```julia
Z, μ = demean(X)
```
"""
function demean(X::AbstractMatrix, m::Nothing)
    d, _ = size(X)
    μ = [mean(skipmissing(view(X, i, :))) for i in 1:d]
    return centralize(X, μ), μ
end

function demean(X::AbstractMatrix, m::AbstractVector)
    return centralize(X, m), m
end

function demean(X::AbstractMatrix, m::Real)
    return centralize(X, fill(m, size(X, 1))), m
end

# ────────────────────────────────────────────────────────────────────────────────
# Core algorithm (Cai, Ma & Wu 2019; Xue & Zou 2023)
# ────────────────────────────────────────────────────────────────────────────────

"""
    fit(HeteroPCAModel, X; rank, maxiter=1_000, abstol=1e-6, α=1.0)

Estimate a HeteroPCAModel model from a `d × n` matrix `X` that **may contain
`missing`**.  Missing values are demeaned and replaced by zero before
computing the sample cross‑product.

`rank` (a positive integer) is the target dimensionality *k*.
`α` is the diagonal update relaxation parameter from Algorithm 2 of
Cai et al. (2019); `α = 1` reproduces the original scheme.
"""
function fit(::Type{HeteroPCAModel}, X::AbstractMatrix{T};
    rank::Integer,
    maxiter::Integer=1_000,
    abstol::Real=1e-6,
    mean=nothing,
    α::Real=1.0) where T<:Real

    d, n = size(X)
    Z, μ = demean(X, mean)

    Σ = (Z * Z') / (n - 1)             # sample cross‑product
    M = Σ - Diagonal(diag(Σ))        # off‑diag part
    U = Matrix{T}(undef, d, rank)                # preallocation
    S = Vector{T}(undef, rank)
    M̃ = similar(M)                     # pre-define M̃ outside the loop
    iter = 0
    converged = false
    while iter < maxiter
        F = svd(M; full=false)           # Truncated SVD (rank ≥ k)
        U = F.U[:, 1:rank]
        S = F.S[1:rank]
        M̃ = U * Diagonal(S) * U'        # best rank‑k approx (sym)
        err = maximum(abs.((diag(M̃) .- diag(M)) ./ diag(M)))
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
    iter == maxiter && @warn "HeteroPCAModel: not converged after $maxiter iterations"

    proj = U
    prinvars = S
    tprinvar = sum(S)
    diagnoise = diag(Σ) .- diag(M̃)
    # Total variance = common component + idiosyncratic noise
    tvar = tprinvar + sum(diagnoise)

    return HeteroPCAModel(μ, proj, prinvars, tprinvar, tvar, diagnoise, converged, iter)
end

# Predict / reconstruct (parallel to pca.jl) ------------------------------------

"""
    predict(M::HeteroPCAModel, x)

Project `x` (vector or matrix) onto the estimated factor space.
"""
function predict(M::HeteroPCAModel, x::AbstractVecOrMat{T}) where {T<:Real}
    z = centralize(x, M.mean)
    return transpose(M.proj) * z
end

"""
    reconstruct(M::HeteroPCAModel, y)

Inverse transform from factor space back to the original *d*‑space.
"""
function reconstruct(M::HeteroPCAModel, y::AbstractVecOrMat{T}) where {T<:Real}
    return decentralize(M.proj * y, M.mean)
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
    heteropca(X; rank, maxiter=1_000, abstol=1e-6, mean=nothing, α=1.0)

Convenience wrapper for `fit(HeteroPCAModel, X, ...)`.
"""
function heteropca(X::AbstractMatrix{T};
    rank::Integer,
    maxiter::Integer=1_000,
    abstol::Real=1e-6,
    mean=nothing,
    α::Real=1.0) where T<:Real

    return fit(HeteroPCAModel, X;
        rank=rank,
        maxiter=maxiter,
        abstol=abstol,
        mean=mean,
        α=α)
end

end # module