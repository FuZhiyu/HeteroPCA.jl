module HeteroPCA

using LinearAlgebra, Statistics, Random
import Base: size, show
using StatsBase
using StatsBase: CoefTable
import Statistics: mean, var
import StatsAPI: fit, predict, r2
import LinearAlgebra: eigvals, eigvecs

include("utils.jl")
include("heteropcamodel.jl")
include("algorithms.jl")

export HeteroPCAModel, fit, predict, reconstruct,
    projection, principalvars, r2, loadings, noisevars, var, eigvals, eigvecs,
    tprincipalvar, tresidualvar, heteropca, principalratio,
    StandardHeteroPCA, DeflatedHeteroPCA, DiagonalDeletion,
    procrustes_align, matrix_distance, sinθ_distance

end # module