function get_dim(X1::Array{T,3}, X2::Matrix{T}, D::Matrix{T}) where T
    N, J, K1 = size(X1)
    J2, K2 = size(X2)
    N2, L = size(D)
    @assert N==N2 && J==J2 
    return (N=N, J=J, K1=K1, K2=K2, L=L) 
end

function get_kdim(X1::Array{T,3}, X2::Matrix{T}, D::Matrix{T}) where T
    N, J, K1, K2, L = get_dim(X1, X2, D)
    kdim = K1+K2*L+J-1
    return kdim
end

"""
Get random choice set of size NC from total J products

For each individividual, the actual choice y_i must be included in the dataset.
Other choices in the subset are randomly picked from fullset with equal probability
"""
function get_random_set(J::Integer, NC::Integer, Y::Vector{T}) where T<:Integer
    N = length(Y)
    C = Vector{Vector{T}}(undef, N)
    for (i, y) in enumerate(Y)
        subset = sample(1:J, NC; replace=false)
        if !(y in subset)
            subset[1] = y
        end
        C[i] = subset
    end
    return C
end



function initial_guess(m::DCModel)
    if !m.hetero_preference
        init_guess = randn(get_kdim(m.X1, m.X2, m.D))
    end
    return init_guess
end