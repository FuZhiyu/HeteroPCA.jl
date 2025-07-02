"Return A rotated to minimize Frobenius distance to B."
function align_A_to_B(A, B)
    U, S, V = svd(B * A')
    R = U * V'
    return R * A
end

"Frobenius error between predicted and true factors after optimal rotation."
function frobenius_error(F̂, Z)
    F̂_aligned = align_A_to_B(F̂, Z)
    return norm(F̂_aligned - Z) / norm(Z)
end