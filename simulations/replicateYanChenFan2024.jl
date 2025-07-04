##############################################################################
#  Yan, Chen & Fan (2024) – Section 4 simulation helper
#
#  ▸ Simulates xj ~ 𝒩(0, Σstar) with  Σstar = Ustar Ustarᵀ   (Λ = I_r  in the paper)
#  ▸ Adds heteroskedastic noise ηℓj  with variances ωℓ²  ~ 𝒰[0.1ω̄, 2ω̄]
#  ▸ Observes a Bernoulli‑p subset of entries
#  ▸ Provides convenience routines to compute the error metrics used in Figs 1–3
##############################################################################

using Random, LinearAlgebra, Statistics, Distributions
using HeteroPCA: procrustes_align, matrix_distance, row_norm

# ---------- utilities -------------------------------------------------------

"""Return a d×r Haar‑random orthonormal matrix (columns)."""
function rand_orthonormal(d::Int, r::Int; rng=Random.GLOBAL_RNG)
    Q, _ = qr!(randn(rng, d, r))
    return Matrix(Q[:, 1:r])  # make it dense
end

# Use HeteroPCA.procrustes_align instead of local sgn_alignment
sgn_alignment(A, B) = procrustes_align(A, B)[2]  # Return just the rotation matrix R

# Use HeteroPCA.row_norm instead of local row_2inf_norm
row_2inf_norm(M) = row_norm(M; p=2, q=Inf)

# ---------- data generation --------------------------------------------------

"""
    simulate_section4(d, n, r, p, ω̄; rng = Random.GLOBAL_RNG)

Return (Y, Ω, X, Σstar, Ustar, ω_vec)

• Σstar = Ustar Ustarᵀ   with Haar‑random Ustar (Λ = I)
• ω_vec[ℓ]  ~  Uniform(0.1ω̄, 2ω̄)
• ηₗⱼ       ~  𝒩(0, ω_vec[ℓ]²)
• Ω         : Bool mask of observed entries (Bernoulli‑p)
• Y         : observed matrix with unobserved entries set to 0
"""
function simulate_section4(d, n, r, p, ω̄; rng=Random.GLOBAL_RNG)
    # 1. latent low‑rank structure
    Ustar = rand_orthonormal(d, r; rng)         # Ustar  ∈ ℝ^{d×r}
    X = Ustar * randn(rng, r, n)               # columns x_j

    # 2. heteroskedastic noise
    ω_vec = rand(rng, Uniform(0.1ω̄, 2ω̄), d)
    N = ω_vec .* randn(rng, d, n)        # broadcast rowwise std‑dev

    # 3. Bernoulli sampling mask
    Ω = rand(rng, d, n) .< p                # Bool matrix
    Y = zeros(d, n)
    Y[Ω] .= (X.+N)[Ω]                     # observed entries; rest remain 0

    Σstar = Ustar * Ustar'                           # "ground‑truth" covariance (Λ = I)
    return Y, Ω, X, Σstar, Ustar, ω_vec
end

# ---------- error metrics ----------------------------------------------------

"""
    subspace_errors(Û, Ustar)

Return a NamedTuple with the three relative subspace errors used in Yan et al. (2024):

• op  – spectral  ‖ÛR − U⋆‖₂ / ‖U⋆‖₂   (‖U⋆‖₂ = 1)  
• fro – Frobenius ‖·‖_F / ‖U⋆‖_{2,∞}  
• max – row‑ℓ₂‑∞ ‖·‖_{2,∞} / ‖U⋆‖_{2,∞}
"""
function subspace_errors(Û, Ustar)
    return (op=matrix_distance(Û, Ustar; norm=:spectral, align=true, relative=false),
            fro=matrix_distance(Û, Ustar; norm=:fro, align=true, relative=true),
            max=matrix_distance(Û, Ustar; norm=:max, align=true, relative=true))
end
"""
    covariance_errors(Ŝ, Σstar)

Return a NamedTuple of the three relative covariance errors (spectral, Frobenius, entry‑wise ∞‑norm).
"""
function covariance_errors(Ŝ, Σstar)
    return (op=matrix_distance(Ŝ, Σstar; norm=:spectral, align=false, relative=true),
            fro=matrix_distance(Ŝ, Σstar; norm=:fro, align=false, relative=true),
            inf=matrix_distance(Ŝ, Σstar; norm=:inf, align=false, relative=true))
end
# ---------- example ----------------------------------------------------------
##############################################################################
#  Replicate Fig. 2 from Yan, Chen & Fan (2024)
#
#  – simulation_section4, subspace_errors, covariance_errors are the helpers
#    we defined previously;
#  – Algorithm 1 (vanilla SVD) is implemented below exactly as described
#    in the paper an et al. - 2024 - Inference for heteroskedastic PCA with missing data.pdf](file-service://file-HdJ3TikabWmM4nMV9dCTgD);
#  – We sweep p over the same grid used in the paper and average over 200 trials,
#    then plot the three error metrics for both SVD and HeteroPCA Yan et al. - 2024 - Inference for heteroskedastic PCA with missing data.pdf](file-service://file-HdJ3TikabWmM4nMV9dCTgD).
##############################################################################

using Random, LinearAlgebra, Statistics, ProgressMeter, Plots

# ---------------------------------------------------------------------------
#  1) Vanilla SVD‑based estimator (Algorithm 1, paper §2)  -------------------
# ---------------------------------------------------------------------------

"""
    svd_estimator(Y, p, r)

Return (Û, Ŝ) from Algorithm 1:
  • Û … top‑r left singular vectors of (p⁻¹Y)/√n
  • Ŝ … rank‑r covariance estimate Û Σ̂² Ûᵀ
"""
function svd_estimator(Y, p, r)
    n = size(Y, 2)
    Ys = (Y / p) / √n
    U, s, _ = svd(Ys; full=false)
    Û = U[:, 1:r]
    Σ̂ = Diagonal(s[1:r] .^ 2)
    Ŝ = Û * Σ̂ * Û'
    return Û, Ŝ
end

# ---------------------------------------------------------------------------
#  2) Experiment parameters (Section 4 setup)  -------------------------------
# ---------------------------------------------------------------------------

d, n, r = 100, 2000, 3          # fixed across Fig. 2
ω̄ = 0.05                 # noise level
p_grid = 0.05:0.01:0.3         # sampling rates
R = 200                  # Monte‑Carlo replications (Fig. 2 caption)

# storage: rows = p‑grid, cols = error metrics (spectral, Frobenius/‖·‖₂,∞, max/ℓ₂,∞)
errU_svd = zeros(length(p_grid), 3)
errU_het = zeros(length(p_grid), 3)
errS_svd = zeros(length(p_grid), 3)
errS_het = zeros(length(p_grid), 3)

# ---------------------------------------------------------------------------
#  3) Monte‑Carlo loop  ------------------------------------------------------
# ---------------------------------------------------------------------------

@showprogress 1 "Running simulations…" for (i, p) in enumerate(p_grid)
    for _ in 1:R
        Y, Ω, X, Σstar, Ustar, ω_vec = simulate_section4(d, n, r, p, ω̄)

        # --- SVD
        Û_svd, Ŝ_svd = svd_estimator(Y, p, r)
        eu_svd = subspace_errors(Û_svd, Ustar)
        es_svd = covariance_errors(Ŝ_svd, Σstar)

        # --- HeteroPCA  (your routine)
        Û_het = heteropca(Y, r, demean=false, impute_method=:zero, abstol=1e-4).proj         # assumes your heteropca returns a struct
        Ŝ_het = Û_het * Û_het'             # rank‑r projection (Λ̂=I suffices for error)
        eu_het = subspace_errors(Û_het, Ustar)
        es_het = covariance_errors(Ŝ_het, Σstar)

        # accumulate
        errU_svd[i, :] .+= collect(values(eu_svd))
        errS_svd[i, :] .+= collect(values(es_svd))
        errU_het[i, :] .+= collect(values(eu_het))
        errS_het[i, :] .+= collect(values(es_het))
    end
end

errU_svd ./= R;
errS_svd ./= R;
errU_het ./= R;
errS_het ./= R;

# ---------------------------------------------------------------------------
#  4) Plot (two panels like Fig. 2)  ----------------------------------------
# ---------------------------------------------------------------------------
##
default(markerstrokewidth=1)   # cleaner points

# Panel (a): subspace errors
pltU = plot(title="Relative subspace errors vs. sampling rate p",
    xlabel="p", ylabel="relative error", legend=:topright)

lblsU = ["spectral ‖·‖₂", "Frobenius", "Inf"]
mks = [:circle, :star8, :square]

for j in 1:3
    scatter!(pltU, p_grid, errU_svd[:, j], marker=mks[j], label="SVD – $(lblsU[j])", markerstrokecolor=:red, markercolor=:white)
    scatter!(pltU, p_grid, errU_het[:, j], marker=mks[j], label="HeteroPCA – $(lblsU[j])", markerstrokecolor=:blue, markercolor=:white)
end

# Panel (b): covariance errors
pltS = plot(title="Relative covariance errors vs. sampling rate p",
    xlabel="p", ylabel="relative error", legend=:topright)
lblsS = ["spectral ‖·‖₂", "Frobenius", "Inf"]
mks = [:circle, :star8, :square]

for j in 1:3
    scatter!(pltS, p_grid, errS_svd[:, j], marker=mks[j], label="SVD – $(lblsS[j])", markerstrokecolor=:red, markercolor=:white)
    scatter!(pltS, p_grid, errS_het[:, j], marker=mks[j], label="HeteroPCA – $(lblsS[j])", markerstrokecolor=:blue, markercolor=:white)
end

plot(pltU, pltS, layout=(1, 2), size=(1200, 480))
##
savefig("simulations/figure2_replication.pdf")