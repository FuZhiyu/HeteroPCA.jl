using LinearAlgebra

# =============================================================================
# Procrustes Alignment Functions
# =============================================================================

"""
    procrustes_align(A, B) -> (A_aligned, R)

Find the orthogonal matrix R that minimizes ||AR - B||_F and return both
the aligned matrix A*R and the rotation matrix R.

This replaces: rotate_colspace, sgn_alignment, procrustes_rotation
"""
function procrustes_align(A, B)
    F = svd(A' * B)
    R = F.U * F.Vt
    return A * R, R
end

# Keep rotate_colspace for backward compatibility
"Return A rotated in columns to minimize Frobenius distance to B."
rotate_colspace(A, B) = procrustes_align(A, B)[1]

# =============================================================================
# Norm Functions
# =============================================================================

"""
    row_norm(M; p=2, q=Inf)

Compute the ℓq norm of ℓp norms of rows.
Default (p=2, q=Inf) gives the row-wise ℓ₂-∞ norm.
"""
function row_norm(M; p=2, q=Inf)
    row_norms = [norm(row, p) for row in eachrow(M)]
    return norm(row_norms, q)
end

# =============================================================================
# Distance Functions
# =============================================================================

# Helper for orthonormalizing columns
ortho_cols(A) = Matrix(qr(A).Q)

"""
    sinθ_distance(U, Uhat; norm = :fro)

Compute sin-Θ distance between two subspaces.

# Arguments
- `U`, `Uhat`: Matrices whose columns span the subspaces
- `norm`: Distance norm
  - `:spectral` - Operator 2-norm (spectral norm)
  - `:fro` - Frobenius norm (default)
  - `:max` - Row-wise ℓ₂-∞ norm

# Returns
The sin-Θ distance in the specified norm.
"""
function sinθ_distance(U, Uh; norm=:fro)
    Q1 = ortho_cols(U)
    Q2 = ortho_cols(Uh)
    Δ = Q1 * Q1' - Q2 * Q2'

    if norm === :spectral
        return opnorm(Δ, 2)
    elseif norm === :fro
        return LinearAlgebra.norm(Δ, 2)  # Frobenius norm
    elseif norm === :max
        return row_norm(Δ; p=2, q=Inf)
    else
        # Default to Frobenius for unknown norms (type stable)
        return LinearAlgebra.norm(Δ, 2)
    end
end

"""
    matrix_distance(A, B; norm=:fro, align=false, relative=false)

Compute distance between matrices A and B.

# Arguments
- `A`, `B`: Matrices to compare
- `norm`: Distance norm
  - `:spectral` - Operator 2-norm
  - `:fro` - Frobenius norm (default)
  - `:max` - Row-wise ℓ₂-∞ norm
  - `:inf` - Entry-wise ∞-norm
- `align`: If true, optimally align A to B before computing distance
- `relative`: If true, return relative error (normalized by norm of B)

# Returns
The distance between A and B in the specified norm.
"""
function matrix_distance(A, B; norm=:fro, align=false, relative=false)
    if align
        A_aligned, _ = procrustes_align(A, B)
        Δ = A_aligned - B
    else
        Δ = A - B
    end

    if norm === :spectral
        dist = opnorm(Δ, 2)
        relative && (dist /= opnorm(B, 2))
    elseif norm === :fro
        dist = LinearAlgebra.norm(Δ, 2)
        relative && (dist /= LinearAlgebra.norm(B, 2))
    elseif norm === :max
        dist = row_norm(Δ; p=2, q=Inf)
        relative && (dist /= row_norm(B; p=2, q=Inf))
    elseif norm === :inf
        dist = maximum(abs, Δ)
        relative && (dist /= maximum(abs, B))
    else
        # Default to Frobenius for unknown norms (type stable)
        dist = LinearAlgebra.norm(Δ, 2)
        relative && (dist /= LinearAlgebra.norm(B, 2))
    end

    return dist
end