#!/usr/bin/env julia
#=
Complete Figure 4 replication from the Deflated HeteroPCA paper.
Generates all 6 subplots: (a,b) vs sigma, (c,d) vs kappa, (e,f) vs n
=#

using LinearAlgebra, Random, Statistics, Printf
using Plots, Plots.Measures
using HeteroPCA

# Simulation functions
function procrustes_rotation(U_hat, U_true)
    F = svd(U_hat' * U_true)
    return F.U * F.Vt
end

function compute_errors(U_hat, U_true)
    R = procrustes_rotation(U_hat, U_true)
    U_aligned = U_hat * R
    E = U_aligned - U_true

    spectral_error = opnorm(E, 2)
    two_to_infty_error = sqrt(maximum(sum(E .^ 2, dims=2)))

    return (spectral_error=spectral_error, two_to_infty_error=two_to_infty_error)
end

function run_experiment(n1::Int, n2::Int, r::Int, kappa::Real, sigma::Real, seed::Int)
    Random.seed!(seed)

    # Factor model setup
    λ = n1 / n2 + sqrt(n1 / n2)

    # True subspace
    E1 = randn(n1, r)
    U = svd(E1).U

    # Factor coefficients
    E2 = randn(n2, r)

    # Singular values with condition number
    D = vcat([kappa * λ], fill(λ, r - 1))
    Λ = Diagonal(D)

    # Signal + noise
    X = U * sqrt(Λ) * E2'
    E = zeros(n1, n2)
    for i in 1:n1
        σᵢ = rand() * sigma
        E[i, :] = σᵢ * randn(n2)
    end
    Y = X + E

    results = NamedTuple[]

    # Test all methods
    results = NamedTuple[]

    # Method 1: Deflated HeteroPCA
    model_deflated = heteropca(Y, r; algorithm=DeflatedHeteroPCA(t_block=5))
    U_deflated = projection(model_deflated)
    errors = compute_errors(U_deflated, U)
    push!(results, (method="Deflated HeteroPCA", spectral_error=errors.spectral_error, two_to_infty_error=errors.two_to_infty_error))

    # Method 2: SVD
    F_svd = svd(Y * Y')
    U_svd = F_svd.U[:, 1:r]
    errors = compute_errors(U_svd, U)
    push!(results, (method="SVD", spectral_error=errors.spectral_error, two_to_infty_error=errors.two_to_infty_error))

    # Method 3: Diagonal Deletion
    model_diag = heteropca(Y, r; algorithm=DiagonalDeletion())
    U_diag = projection(model_diag)
    errors = compute_errors(U_diag, U)
    push!(results, (method="Diagonal Deletion", spectral_error=errors.spectral_error, two_to_infty_error=errors.two_to_infty_error))

    # Method 4: Standard HeteroPCA
    model_hetero = heteropca(Y, r; algorithm=StandardHeteroPCA(), maxiter=200)
    U_hetero = projection(model_hetero)
    errors = compute_errors(U_hetero, U)
    push!(results, (method="HeteroPCA", spectral_error=errors.spectral_error, two_to_infty_error=errors.two_to_infty_error))

    return results
end

function run_simulation(param_name, param_values, default_params, seed_range)
    println("Running $(param_name) simulation...")
    all_results = NamedTuple[]

    for seed in seed_range
        for param_val in param_values
            params = merge(default_params, Dict(param_name => param_val))
            results = run_experiment(params[:n1], params[:n2], params[:r], params[:kappa], params[:sigma], seed)

            for result in results
                push!(all_results, merge(result, (seed=seed, param_name => param_val)))
            end
        end
    end

    return all_results
end

function compute_summary(results, param_name)
    method_names = ["Deflated HeteroPCA", "SVD", "Diagonal Deletion", "HeteroPCA"]
    param_values = unique([r[param_name] for r in results])

    summary_stats = NamedTuple[]
    for method in method_names
        for param_val in param_values
            method_results = filter(r -> r.method == method && r[param_name] == param_val, results)
            if !isempty(method_results)
                spectral_errors = [r.spectral_error for r in method_results]
                two_to_infty_errors = [r.two_to_infty_error for r in method_results]

                push!(summary_stats, (
                    method=method,
                    param_name => param_val,
                    spectral_error_mean=mean(spectral_errors),
                    two_to_infty_error_mean=mean(two_to_infty_errors)
                ))
            end
        end
    end
    return summary_stats
end

function create_subplot(summary_stats, param_name, param_label, title_prefix1, title_prefix2=title_prefix1, log_scale=false)
    # Match original figure colors and styles exactly
    colors = [:blue, :red, :orange, :black]
    markers = [:utriangle, :circle, :cross, :diamond]
    linestyles = [:solid, :dash, :solid, :dash]
    method_names = ["Deflated HeteroPCA", "SVD", "Diagonal Deletion", "HeteroPCA"]

    # Spectral error plot
    p1 = plot(title="$title_prefix1 ℓ₂ error",
        xlabel=param_label,
        ylabel="ℓ₂ error",
        legend=:topleft, size=(300, 250))

    if log_scale
        plot!(p1, xscale=:log10)
    end

    for (i, method) in enumerate(method_names)
        method_data = filter(r -> r.method == method, summary_stats)
        if !isempty(method_data)
            param_vals = [r[param_name] for r in method_data]
            errors = [r.spectral_error_mean for r in method_data]
            sort_idx = sortperm(param_vals)

            label = method == "Deflated HeteroPCA" ? "Deflated-HeteroPCA" :
                    method == "SVD" ? "Vanilla SVD" :
                    method == "Diagonal Deletion" ? "Diagonal-deleted PCA" : method

            plot!(p1, param_vals[sort_idx], errors[sort_idx],
                label=label, color=colors[i], marker=markers[i],
                linestyle=linestyles[i], linewidth=2, markersize=3)
        end
    end

    # Two-to-infinity error plot
    p2 = plot(title="$title_prefix2 ℓ₂,∞ error",
        xlabel=param_label,
        ylabel="ℓ₂,∞ error",
        legend=:topleft, size=(300, 250))

    if log_scale
        plot!(p2, xscale=:log10)
    end

    for (i, method) in enumerate(method_names)
        method_data = filter(r -> r.method == method, summary_stats)
        if !isempty(method_data)
            param_vals = [r[param_name] for r in method_data]
            errors = [r.two_to_infty_error_mean for r in method_data]
            sort_idx = sortperm(param_vals)

            label = method == "Deflated HeteroPCA" ? "Deflated-HeteroPCA" :
                    method == "SVD" ? "Vanilla SVD" :
                    method == "Diagonal Deletion" ? "Diagonal-deleted PCA" : method

            plot!(p2, param_vals[sort_idx], errors[sort_idx],
                label=label, color=colors[i], marker=markers[i],
                linestyle=linestyles[i], linewidth=2, markersize=3)
        end
    end

    return p1, p2
end

println("=== COMPLETE FIGURE 4 REPLICATION ===")

# Simulation parameters (reduced for speed)
r = 3
seed_range = 2023:2073

# Create results directory
mkpath("results")

# Simulation 1: vs sigma (plots a,b)
println("\n1/3: Sigma variation simulation...")
sigma_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
default_params_sigma = (n1=100, n2=1000, r=r, kappa=100, sigma=0.5)
results_sigma = run_simulation(:sigma, sigma_values, default_params_sigma, seed_range)
summary_sigma = compute_summary(results_sigma, :sigma)
p1a, p1b = create_subplot(summary_sigma, :sigma, "ω: noise level", "(a) κₚc = 100, n = 1,000,", "(b) κₚc = 100, n = 1,000,")

# Simulation 2: vs kappa (plots c,d)  
println("\n2/3: Kappa variation simulation...")
kappa_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
default_params_kappa = (n1=100, n2=1000, r=r, kappa=10, sigma=1.0)
results_kappa = run_simulation(:kappa, kappa_values, default_params_kappa, seed_range)
summary_kappa = compute_summary(results_kappa, :kappa)
p1c, p1d = create_subplot(summary_kappa, :kappa, "κₚc: condition number", "(c) ω = 1, n = 1,000,", "(d) ω = 1, n = 1,000,", true)

# Simulation 3: vs n (plots e,f)
println("\n3/3: Sample size variation simulation...")
n_values = [100, 200, 300, 500, 700, 1000, 2000, 3000, 5000, 7000, 10000]
default_params_n = (n1=100, n2=1000, r=r, kappa=100, sigma=1.0)
results_n = run_simulation(:n2, n_values, default_params_n, seed_range)
summary_n = compute_summary(results_n, :n2)
p1e, p1f = create_subplot(summary_n, :n2, "n: sample size", "(e) ω = 1, κₚc = 100,", "(f) ω = 1, κₚc = 100,", true)

# Combine all subplots into Figure 4
println("\nCreating Figure 4...")
complete_figure = plot(p1a, p1b, p1c, p1d, p1e, p1f,
    layout=(2, 3),
    size=(1200, 900),
    margin=5mm,
    plot_title="Figure 4: Estimation errors under the factor model")

# Save complete figure
savefig(complete_figure, "simulations/ZhouChen2025_figure4_complete_replication.png")
