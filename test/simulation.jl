###############################################################################
# Simulating PCA under the generalised spiked–covariance model
#
#   Yk = Xk + εk ,           k = 1,…,n
#   Xk  ~  N(0, Σ₀)          (Σ₀  =  U diag(1,…,r) Uᵀ )
#   εk  ~  N(0, diag(σ₁²,…,σp²))         iid across k
#
#   Goal : estimate the r‑dimensional principal sub‑space U
#          and report the sin‑Θ distance between  Û  and  U.
#
#   Methods compared :  HeteroPCA()   vs   Regular SVD
###############################################################################
using LinearAlgebra, Random, Statistics, StatsBase
using HeteroPCA           # ← your source file must be on LOAD_PATH
using ProgressMeter
using Plots
using Distributions
# using Distributions: MvNormal
# --------------------- helper: sin–Theta distance ----------------------------
"""
    sintheta_distance(U, Uhat)

Spectral sin–Θ distance ‖P_U − P_Û‖₂ between two r‑dimensional
sub‑spaces with orthonormal bases `U` and `Uhat`.
"""
function sintheta_distance(U::AbstractMatrix, Uh::AbstractMatrix)
    Pu = U * U'
    Puh = Uh * Uh'
    return opnorm(Pu - Puh)           # largest singular value ‖·‖₂
end

# --------------------- simulation parameters ---------------------------------
p = 30                 # ambient dimension
ns = 60:60:600          # sample‑size grid
r_vals = (3, 5)             # target ranks (panels)
nrep = 200                # Monte‑Carlo replications
rng = MersenneTwister(42)

# storage :  Dict {r  ↦  Dict(method => n×nrep matrix of losses)}
losses = Dict{Int,Dict{String,Matrix{Float64}}}()
for r in r_vals
    losses[r] = Dict(
        "HeteroPCA" => similar(zeros(length(ns), nrep)),
        "Regular SVD" => similar(zeros(length(ns), nrep)),
    )
end

# --------------------- main Monte‑Carlo loop ---------------------------------
@showprogress 1 "Running Monte‑Carlo..." for rep in 1:nrep
    for (i, n) in enumerate(ns)
        # ---------- generate a fresh (U, Σ₀, σ²) per replication -------------
        # step 1: heteroskedastic weights and noise     wᵢ, σᵢ²  ~ Unif(0,1)
        w = rand(rng, p)
        σ = rand(rng, p)
        σ2 = σ .^ 2

        ε = randn(rng, p, n) .* σ               # p×n  iid   (pre‑sample once)

        for r in r_vals
            # U_r = U[:, 1:r]                            # ground‑truth sub‑space
            U0_r = randn(rng, p, r)
            U_r = Matrix(qr(diagm(w) * U0_r).Q)           # U_r ∈ O_{p,r}
            # -- generate rank‑r signal X and corresponding observations Y ----------
            Λ_r = U_r * Diagonal(sqrt.(1:r))                # p × r  loadings
            X = Λ_r * randn(rng, r, n)                   # p × n  low‑rank factors
            Y = X + ε                                    # p × n  observations

            # --------------------------------- HeteroPCA --------------------
            M̂ = fit(HeteroPCAModel, Y; rank=r)          # mean‑free already
            Û_hpca = projection(M̂)                     # p×r
            d_hpca = sintheta_distance(U_r, Û_hpca)
            losses[r]["HeteroPCA"][i, rep] = d_hpca

            # -------------------------------- Regular SVD --------------------
            C = (Y * Y') / (n - 1)                 # sample covariance
            F = eigen(Symmetric(C))
            Û_svd = F.vectors[:, end-r+1:end]         # top‑r eigenvectors
            d_svd = sintheta_distance(U_r, Û_svd)
            losses[r]["Regular SVD"][i, rep] = d_svd
        end
    end
end

# --------------------- summarise & plot --------------------------------------
# palette = [:red, :olive, :teal]
##
plt = plot(layout=(length(r_vals), 1), legend=false, size=(700, 800))
for (panel, r) in enumerate(r_vals)
    ngrid = collect(ns)
    for (j, method) in enumerate(["HeteroPCA", "Regular SVD"])
        lossmat = losses[r][method]
        means = vec(mean(lossmat, dims=2))
        sds = vec(std(lossmat, dims=2))
        plot!(plt[panel], ngrid, means;
            ribbon=sds, label=method,
            linewidth=2, marker=:circle)
    end
    ylabel!(plt[panel], "Sin–Θ Distance")
    xlabel!(plt[panel], "n")
    title!(plt[panel], "p = 30,  r = $r")
    # grid!(plt[panel], :y)
end
plot!(plt, legend=:right)
##
# savefig(plt, "fig1_generalised_spiked_covariance.pdf")
# println("✓  Simulation finished – figure saved to fig1_generalised_spiked_covariance.pdf")