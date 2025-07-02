using Test
using Random
using LinearAlgebra
using HeteroPCA

# deterministic RNG
Random.seed!(20240518)

# --------------------------------------------------------------------------
# helper utilities
# --------------------------------------------------------------------------

"Return a d×r Haar–distributed orthonormal matrix."
rand_orthonormal(d::Int, r::Int) = Matrix(qr(randn(d, r)).Q[:, 1:r])

"""
    simulate(d, n, r; p = 1, ω̄ = 0)

Generate a rank‑r signal plus heteroskedastic noise and an optional
Bernoulli‑p missing mask (à la Yan‑Chen‑Fan §4).

Returns `(Y, Ustar, Z)` where `Y` may contain `missing`.
"""
function simulate(d, n, r; p=1.0, ω̄=0.0)
    Ustar = rand_orthonormal(d, r)
    Z = randn(r, n)
    X = Ustar * Z

    ω_vec = ω̄ == 0 ? zeros(d) : (0.1 .+ 1.9 .* rand(d)) .* ω̄
    N = ω_vec .* randn(d, n)

    if p == 1
        return X + N, Ustar, Z
    else
        mask = rand(d, n) .< p
        Y = Matrix{Union{Missing,Float64}}(missing, d, n)
        Y[mask] .= (X+N)[mask]
        return Y, Ustar, Z
    end
end


# --------------------------------------------------------------------------
# global dims used in all tests
# --------------------------------------------------------------------------
d = 30
n = 500
r = 3

# --------------------------------------------------------------------------
# 1) noiseless, fully observed
# --------------------------------------------------------------------------
@testset "HeteroPCA – noiseless, no missing" begin
    Y, Ustar, Z = simulate(d, n, r; p=1.0, ω̄=0.0)
    model = heteropca(Y, r; demean=false, abstol=1e-10)

    @test frobenius_error(projection(model), Ustar) ≤ 1e-8

    # test factor recovery
    F̂ = predict(model, Y)
    @test frobenius_error(F̂, Z) ≤ 1e-8

    # round‑trip a few columns through predict / reconstruct
    for j in 1:5
        ẑ = predict(model, Y[:, j])
        ŷ = reconstruct(model, ẑ)
        @test norm(ŷ - Y[:, j]) ≤ 1e-8
    end
end


# --------------------------------------------------------------------------
# 2) small noise, no missing (n_large, 40 replications)
# --------------------------------------------------------------------------
@testset "HeteroPCA – small noise, no missing (avg over 40)" begin
    n_large = 2000
    R = 40
    sub_err = 0.0
    fac_err = 0.0
    for _ in 1:R
        Y, Ustar, Z = simulate(d, n_large, r; p=1.0, ω̄=0.02)
        m = heteropca(Y, r; demean=false, impute_method=:zero)
        sub_err += frobenius_error(projection(m), Ustar)
        fac_err += frobenius_error(predict(m, Y), Z)
    end
    sub_err /= R
    fac_err /= R
    @test sub_err ≤ 0.01
    @test fac_err ≤ 0.05
end

# --------------------------------------------------------------------------
# 3) missing values, two imputation schemes (avg over 40)
# --------------------------------------------------------------------------
@testset "HeteroPCA – p = 0.5, small noise (avg over 40)" begin
    n_large = 2000
    R = 40
    p = 0.5
    sub_zero = 0.0
    fac_zero = 0.0
    sub_pair = 0.0
    fac_pair = 0.0
    for _ in 1:R
        Y, Ustar, Z = simulate(d, Int(n_large / p), r; p=p, ω̄=0.02)

        m0 = heteropca(Y, r; demean=true, impute_method=:zero)
        sub_zero += frobenius_error(projection(m0), Ustar)
        fac_zero += frobenius_error(predict(m0, Y), Z)

        mp = heteropca(Y, r; demean=true, impute_method=:pairwise)
        sub_pair += frobenius_error(projection(mp), Ustar)
        fac_pair += frobenius_error(predict(mp, Y), Z)
    end
    sub_zero /= R
    fac_zero /= R
    sub_pair /= R
    fac_pair /= R

    @test sub_zero ≤ 0.05
    @test fac_zero ≤ 0.05
    @test sub_pair ≤ 0.05
    @test fac_pair ≤ 0.05
end

@testset "DeflatedHeteroPCA – p = 0.5, small noise (avg over 40)" begin
    n_large = 2000
    R = 40
    p = 0.5
    sub_zero = 0.0
    fac_zero = 0.0
    sub_pair = 0.0
    fac_pair = 0.0
    for _ in 1:R
        Y, Ustar, Z = simulate(d, Int(n_large / p), r; p=p, ω̄=0.02)

        m0 = heteropca(Y, r; demean=true, impute_method=:zero, algorithm=DeflatedHeteroPCA())
        sub_zero += frobenius_error(projection(m0), Ustar)
        fac_zero += frobenius_error(predict(m0, Y), Z)

        mp = heteropca(Y, r; demean=true, impute_method=:pairwise, algorithm=DeflatedHeteroPCA())
        sub_pair += frobenius_error(projection(mp), Ustar)
        fac_pair += frobenius_error(predict(mp, Y), Z)
    end
    sub_zero /= R
    fac_zero /= R
    sub_pair /= R
    fac_pair /= R

    @test sub_zero ≤ 0.05
    @test fac_zero ≤ 0.05
    @test sub_pair ≤ 0.05
    @test fac_pair ≤ 0.05
end
