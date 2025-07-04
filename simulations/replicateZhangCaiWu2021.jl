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
using HeteroPCA
using HeteroPCA: sinθ_distance, ortho_cols
using ProgressMeter
using Plots
using Distributions

# ───────────────────────── reusable helpers ──────────────────────────

# ortho_cols imported from HeteroPCA

"""
    generate_dataset(p, n, r; α = 0.0, θ = 0.0, rng = Random.GLOBAL_RNG)

Generate a single synthetic data set under the Zhang-Cai-Wu  model:

* `p`          : ambient dimension  
* `n`          : sample size  
* `r`          : true rank  
* `α`          : heteroskedasticity exponent (α = 0 ⇒ homoskedastic)  
* `θ`          : probability that **each** entry is *observed*
                 (θ = 0 ⇒ fully observed;  θ = 0.1 ⇒ 10 % observed)  

Returns `(U_true, Y)` where `U_true` is `p×r` and `Y` is either a
`Float64` matrix (no missingness) or a matrix of `Union{Float64,Missing}`.
"""
function generate_dataset(p, n, r;
    α::Union{Real,Nothing}=nothing,
    θ::Real=0.0,
    rng=Random.GLOBAL_RNG)

    w = rand(rng, p)                          # heteroskedastic weights
    U0_r = randn(rng, p, r)
    U_r = ortho_cols(diagm(w) * U0_r)           # ground‑truth sub‑space

    # noise variances σ_k²  per Section 4.3 of the paper
    if α === nothing                     # original setting: σ ∼ Unif(0,1)
        σ = rand(rng, p)
        σ2 = σ .^ 2
    else                                 # Section 4.3 setting with exponent α
        v = rand(rng, p)
        σ2 = 0.1 * p * v .^ α ./ sum(v .^ α)
        σ = sqrt.(σ2)
    end

    ε = randn(rng, p, n) .* σ                 # idiosyncratic noise
    Λ_r = U_r * Diagonal(sqrt.(1:r))            # loadings
    X = Λ_r * randn(rng, r, n)                # low‑rank signal
    Y = X + ε

    if θ > 0                                     # inject missingness
        mask = rand(rng, p, n) .< θ
        Ymiss = Array{Union{Float64,Missing}}(undef, p, n)
        for idx in eachindex(Y)
            Ymiss[idx] = mask[idx] ? Y[idx] : missing
        end
        return U_r, Ymiss
    else
        return U_r, Y
    end
end

# ───────────────────────── heteroskedastic rectangular SVD helper ────────────
function generate_matrix_hetero_svd(p1::Int, p2::Int, r::Int;
    σ0::Float64=0.2,
    θ::Float64=0.0,
    rng::AbstractRNG=Random.GLOBAL_RNG)

    # low‑rank factors
    U0 = randn(rng, p1, r)
    V0 = randn(rng, p2, r)
    w = rand(rng, p1)
    U = ortho_cols(Diagonal(w .^ 4) * U0)   # row tilt
    V = ortho_cols(V0)

    X = (p1 * p2)^(1 / 4) * U * Diagonal(1:r) * V'  # signal matrix

    # heteroskedastic noise
    v1 = rand(rng, p1)
    v2 = rand(rng, p2)
    Σij = (v1 .^ 4) * (v2 .^ 4)'                     # outer product
    # E = randn(rng, p1, p2) .* (σ0 .* sqrt.(Σij))
    E = randn(rng, p1, p2) .* (σ0 .* Σij)

    Y = X + E

    # optional missingness
    if θ > 0
        mask = rand(rng, p1, p2) .< θ
        Ymiss = Array{Union{Float64,Missing}}(undef, p1, p2)
        for idx in eachindex(Y)
            Ymiss[idx] = mask[idx] ? Y[idx] : missing
        end
        return U, Ymiss
    else
        return U, Y
    end
end

"""
    estimate_subspaces(Y, r)

Return a `Dict(method => Û)` with the sub‑space estimates obtained from  
*HeteroPCA* and *regular SVD*.
"""
function estimate_subspaces(Y, r; abstol=1e-4, kwargs...)
    Û = Dict{String,Matrix{Float64}}()

    # ----- HeteroPCA ---------------------------------------------------
    M̂ = fit(HeteroPCAModel, Y, r; abstol=abstol, kwargs...)
    Û["HeteroPCA"] = projection(M̂)

    # ----- Regular SVD -------------------------------------------------
    d, n = size(Y)
    Yfilled = coalesce.(Y, zero(eltype(Y)))
    C = (Yfilled * Yfilled') / (n - 1)                      # sample covariance
    F = eigen(Symmetric(C))
    Û["Regular SVD"] = F.vectors[:, end-r+1:end]

    return Û
end

# sinθ_distance imported from HeteroPCA

"""
    run_simulation(ns, nrep, r_vals, p; α = 0.0, θ = 0.0, rng)

Monte‑Carlo driver that returns the 3‑level `losses` dictionary:
`losses[r][method][i, rep]`  where  `i` iterates over `ns`.
"""
function run_simulation(ns::AbstractVector{<:Integer},
    nrep::Integer,
    r_vals::AbstractVector{<:Integer},
    p::Integer;
    α::Union{Float64,Nothing}=nothing,
    θ::Float64=0.0,
    rng::AbstractRNG=Random.GLOBAL_RNG)

    losses = Dict{Int,Dict{String,Matrix{Float64}}}()
    for r in r_vals
        losses[r] = Dict("HeteroPCA" => zeros(length(ns), nrep),
            "Regular SVD" => zeros(length(ns), nrep))
    end

    @showprogress 1 "Running Monte‑Carlo..." for rep in 1:nrep
        for (i, n) in enumerate(ns)
            for r in r_vals
                U_true, Y = generate_dataset(p, n, r; α=α, θ=θ, rng=rng)
                Ûs = estimate_subspaces(Y, r)
                for (method, Û) in Ûs
                    losses[r][method][i, rep] =
                        sinθ_distance(U_true, Û; norm=:spectral)
                end
            end
        end
    end
    return losses
end

# ==============================================================================
# Convenience wrappers that reproduce Figures 1, 2, and 5 of Zhang-Cai-Wu   
# ==============================================================================




##
# ==============================================================================
# Convenience wrappers that reproduce Figures 1, 2, and 5 of Zhang-Cai-Wu   
# ==============================================================================

"""
    figure1_sample_size()

Reproduce Fig. 1: sin‑Θ loss vs. sample size *n* (Section 4.1).
Saves `fig1_generalised_spiked_covariance.pdf`.
"""
function figure1_sample_size(; rng=MersenneTwister(42))
    p, r_vals, ns, α, θ, nrep = 30, [3, 5], 60:60:600, nothing, 0.0, 200
    losses = run_simulation(ns, nrep, r_vals, p; α=α, θ=θ, rng=rng)

    plt = plot(layout=(length(r_vals), 1), legend=:right, size=(730, 800))
    for (panel, r) in enumerate(r_vals)
        means = [mean(losses[r][m], dims=2)[:] for m in keys(losses[r])]
        sds = [std(losses[r][m], dims=2)[:] for m in keys(losses[r])]
        for (j, m) in enumerate(["HeteroPCA", "Regular SVD"])
            plot!(plt[panel], ns, means[j]; ribbon=sds[j], label=m,
                marker=:circle, linewidth=2)
        end
        title!(plt[panel], "p = $p,  r = $r")
        xlabel!(plt[panel], "n")
        ylabel!(plt[panel], "Sin–Θ Distance")
    end
    savefig(plt, "simulations/fig1_generalised_spiked_covariance.pdf")
    @info "Fig. 1 saved → fig1_generalised_spiked_covariance.pdf"
    return plt
end
figure1_sample_size()


"""
    figure2_alpha()

Reproduce Fig. 2: sin‑Θ loss vs. heteroskedasticity exponent α (Section 4.3).
Saves `fig2_alpha_heteroskedasticity.pdf`.
"""
function figure2_alpha(; rng=MersenneTwister(43))
    p_vals = [50, 200]
    n_vals = [30, 400]
    r = 5
    alphas = 0:1:10
    methods = ["HeteroPCA", "Regular SVD"]
    nrep = 200

    plt = plot(layout=(2, 1), legend=:right, size=(700, 800))

    for (panel, (p, n)) in enumerate(zip(p_vals, n_vals))
        losses = Dict(m => zeros(length(alphas), nrep) for m in methods)
        for (i, α) in enumerate(alphas), rep in 1:nrep
            U_true, Y = generate_dataset(p, n, r; α=α, θ=0.0, rng=rng)
            Û = estimate_subspaces(Y, r)
            for m in methods
                losses[m][i, rep] = sinθ_distance(U_true, Û[m])
            end
        end
        for m in methods
            μ = mean(losses[m], dims=2)[:]
            σ = std(losses[m], dims=2)[:]
            plot!(plt[panel], alphas, μ; ribbon=σ, label=m,
                marker=:circle, lw=2)
        end
        xlabel!(plt[panel], "alpha")
        ylabel!(plt[panel], "Sin–Θ Distance")
        title!(plt[panel], "p = $p, n = $n, r = $r")
    end
    savefig(plt, "simulations/fig2_alpha_heteroskedasticity.pdf")
    @info "Fig. 2 saved → fig2_alpha_heteroskedasticity.pdf"
    return plt
end
figure2_alpha()


function figure3_alpha(; rng=MersenneTwister(43))
    p_vals = [50, 200]
    n_vals = [200, 1000]
    r = 3
    alphas = 0:0.5:2
    methods = ["HeteroPCA", "Regular SVD"]
    nrep = 100

    plt = plot(layout=(length(p_vals), 1), legend=:right, size=(700, 800))

    for (panel, (p, n)) in enumerate(zip(p_vals, n_vals))
        losses = Dict(m => zeros(length(alphas), nrep) for m in methods)
        @showprogress 1 "Panel $(panel): Monte‑Carlo ..." for (i, α) in enumerate(alphas), rep in 1:nrep
            U_true, Y = generate_matrix_hetero_svd(p, n, r; θ=0.0, rng=rng, σ0=α)
            Û = estimate_subspaces(Y, r)
            for m in methods
                losses[m][i, rep] = sinθ_distance(U_true, Û[m])
            end
        end
        for m in methods
            μ = mean(losses[m], dims=2)[:]
            σ = std(losses[m], dims=2)[:]
            plot!(plt[panel], alphas, μ; ribbon=σ, label=m,
                marker=:circle, lw=2)
        end
        xlabel!(plt[panel], "alpha")
        ylabel!(plt[panel], "Sin–Θ Distance")
        title!(plt[panel], "p = $p, n = $n, r = $r")
    end
    savefig(plt, "simulations/fig3_sigma0.pdf")
    @info "Fig. 3 saved → fig3_sigma0.pdf"
    return plt
end
figure3_alpha()
