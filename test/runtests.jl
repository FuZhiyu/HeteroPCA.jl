using HeteroPCA
using Test
using Random
using LinearAlgebra

@testset "HeteroPCA.jl" begin
    # Set random seed for reproducibility
    Random.seed!(123)

    # Create synthetic data with known structure
    n = 100  # observations
    p = 20   # variables

    # Create data with a few dominant components
    X = zeros(p, n)

    # Component 1: affects variables 1-5
    for i in 1:5
        X[i, :] = randn(n) * sqrt(10.0) + i
    end

    # Component 2: affects variables 6-10
    for i in 6:10
        X[i, :] = randn(n) * sqrt(5.0) - i
    end

    # Component 3: affects variables 11-15
    for i in 11:15
        X[i, :] = randn(n) * sqrt(2.0) + 2i
    end

    # Add random noise to all variables
    X += randn(p, n)

    # Fit the HeteroPCA model with 3 components
    model = fit(HeteroPCAResult, X, maxoutdim=3)

    # Check that the model has the expected properties
    @test size(model) == (p, 3)
    @test length(model.principalvars) == 3
    @test model.principalvars[1] > model.principalvars[2] > model.principalvars[3]
    @test model.converged == true

    # Test projection and reconstruction
    Y = predict(model, X)
    @test size(Y) == (3, n)

    X_reconstructed = reconstruct(model, Y)
    @test size(X_reconstructed) == size(X)

    # Verify that the reconstruction error is reasonable
    @test norm(X_reconstructed - X) / norm(X) < 0.3  # Allow up to 30% reconstruction error

    # Test other utility functions
    @test length(mean(model)) == p
    @test size(projection(model)) == (p, 3)
    @test length(principalvars(model)) == 3
    @test tprincipalvar(model) > 0
    @test var(model) > 0
    @test 0 <= principalratio(model) <= 1

    # Display the model
    println(model)

    # Test with a single vector
    x_single = X[:, 1]
    y_single = predict(model, x_single)
    @test length(y_single) == 3
    x_reconstructed = reconstruct(model, y_single)
    @test length(x_reconstructed) == p
end
