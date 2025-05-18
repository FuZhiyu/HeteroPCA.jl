using HeteroPCA, Test, LinearAlgebra, Random, Statistics

# @testset "HeteroPCA Utility Functions" begin
# Set random seed for reproducibility
Random.seed!(123)

# Create synthetic data with known structure
d = 10  # dimensions/variables
n = 50  # observations
k = 3   # number of factors/components

# Generate synthetic data with known factors
factors = randn(k, n)
factorloadings = randn(d, k)
noise_vars = abs.(randn(d)) .+ 0.5  # Heteroscedastic noise variance

# Generate data matrix with heteroscedastic noise
X = factorloadings * factors
for i in 1:d
    X[i, :] .+= sqrt(noise_vars[i]) * randn(n)
end

# Insert some missing values randomly
X_with_missing = convert(Matrix{Union{Missing,Float64}}, copy(X))
miss_mask = rand(d, n) .< 0.1
X_with_missing[miss_mask] .= missing

# Fit HeteroPCA model
model = fit(HeteroPCAModel, X, k)
model_with_missing = fit(HeteroPCAModel, X_with_missing, k)

# Test predict function
@testset "predict" begin
    # Test predict with full matrix
    Y = predict(model, X)
    @test size(Y) == (k, n)

    # Test predict with a single vector
    y_single = predict(model, X[:, 1])
    @test size(y_single) == (k,)
    @test Y[:, 1] ≈ y_single

    # Test predict with missing data (model handles missing values internally)
    X_filled = copy(X_with_missing)
    for i in 1:d
        col_mean = mean(skipmissing(X_with_missing[i, :]))
        for j in 1:n
            if ismissing(X_filled[i, j])
                X_filled[i, j] = col_mean
            end
        end
    end
    Y_missing = predict(model_with_missing, X_filled)
    @test size(Y_missing) == (k, n)

    # Test that predict handles demeaning correctly
    X_not_centered = X .+ randn(d)
    model_not_centered = fit(HeteroPCAModel, X_not_centered, k)
    Y_not_centered = predict(model_not_centered, X_not_centered)
    @test size(Y_not_centered) == (k, n)

    # Test mathematical correctness of predict function
    # Manual prediction should match the predict function
    X_centered = X .- mean(model)
    Y_manual = transpose(projection(model)) * X_centered
    @test Y_manual ≈ Y

end

@testset "reconstruct" begin
    # Test basic reconstruction
    Y = predict(model, X)
    X_reconstructed = reconstruct(model, Y)
    @test size(X_reconstructed) == size(X)

    # Test that reconstruction preserves the column means
    @test vec(mean(X_reconstructed, dims=2)) ≈ vec(mean(X, dims=2)) atol = 1e-10

    # Test reconstruction with a single vector
    y_single = predict(model, X[:, 1])
    x_reconstructed = reconstruct(model, y_single)
    @test size(x_reconstructed) == (d,)
    @test X_reconstructed[:, 1] ≈ x_reconstructed

    # Test reconstruction with missing data
    X_filled = copy(X_with_missing)
    for i in 1:d
        col_mean = mean(skipmissing(X_with_missing[i, :]))
        for j in 1:n
            if ismissing(X_filled[i, j])
                X_filled[i, j] = col_mean
            end
        end
    end
    Y_missing = predict(model_with_missing, X_filled)
    X_missing_reconstructed = reconstruct(model_with_missing, Y_missing)
    @test size(X_missing_reconstructed) == size(X_with_missing)

    # Test mathematical correctness of reconstruct function
    # Manual reconstruction should match the reconstruct function
    X_manual = projection(model) * Y .+ mean(model)
    @test X_manual ≈ X_reconstructed

    # Verify that projection⋅predict + mean = reconstruct
    @test projection(model) * predict(model, X) .+ mean(model) ≈ reconstruct(model, predict(model, X))

    # Verify that predict⋅reconstruct is not the identity (dimension reduction)
end

@testset "Other utility functions" begin
    # Test size function
    @test size(model) == (d, k)

    # Test mean function
    @test length(mean(model)) == d
    @test mean(model) ≈ vec(mean(X, dims=2))

    # Test projection function
    @test size(projection(model)) == (d, k)
    @test projection(model) == eigvecs(model)

    # Test principalvars and eigvals
    @test length(principalvars(model)) == k
    @test principalvars(model) == eigvals(model)
    @test all(principalvars(model) .> 0)
    @test principalvars(model)[1] >= principalvars(model)[end]  # Sorted in descending order

    # Test tprincipalvar and tresidualvar
    @test tprincipalvar(model) ≈ sum(principalvars(model))
    @test tprincipalvar(model) + tresidualvar(model) ≈ var(model)

    # Test r2 function
    @test 0 <= r2(model) <= 1
    @test r2(model) ≈ tprincipalvar(model) / var(model)
    @test r2(model) == principalratio(model)
    # Test loadings
    ldgs = loadings(model)
    @test size(ldgs) == (d, k)
    @test ldgs ≈ projection(model) * Diagonal(sqrt.(principalvars(model)))

    # Test noisevars
    @test length(noisevars(model)) == d
    @test all(noisevars(model) .>= 0)  # Noise variances should be non-negative
end

@testset "Advanced functionality" begin
    # Test that increasing rank preserves more variance
    model_rank1 = fit(HeteroPCAModel, X, 1)
    model_rank2 = fit(HeteroPCAModel, X, 2)
    @test r2(model_rank1) <= r2(model_rank2)

    # Test that model converges with different settings
    model_tight = fit(HeteroPCAModel, X, k, abstol=1e-8, maxiter=2000)
    @test model_tight.converged

    # Test the heteropca convenience function
    model_hetero = heteropca(X, k)
    @test size(model_hetero) == (d, k)
    @test length(principalvars(model_hetero)) == k

    # Test with different imputation methods
    model_zero = fit(HeteroPCAModel, X_with_missing, k, impute_method=:zero)
    @test size(model_zero) == (d, k)

    # Test with no demeaning
    model_no_demean = fit(HeteroPCAModel, X, k, demean=false)
    @test all(model_no_demean.mean .== 0)
end
