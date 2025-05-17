


function heteroPCA(y::Matrix{T}, rank; maxiter=1000, abstol=1e-6, α=1) where T<:AbstractFloat
    Σ = y * y' / (T - 1)
    M = Σ - Diagonal(diag(Σ))
    iter = 0
    local U, S, V, iter
    while iter < maxiter
        U, S, V = tsvd(M, rank, tolreorth=0.0)
        M̃ = U * Diagonal(S) * V'
        err = norm((diag(M̃) - diag(M)) ./ diag(M), Inf)
        println(err)
        if err < abstol
            break
        end
        for i in 1:N
            M[i, i] = α * M̃[i, i] + (1 - α) * M[i, i]
        end
        iter += 1
    end
    if iter == maxiter
        @error "not converged"
    end
    return U, S, V, iter
end