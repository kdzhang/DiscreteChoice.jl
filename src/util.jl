function get_dim(X1::Array{T,3}, X2::Matrix{T}, D::Matrix{T}) where T
    N, J, K1 = size(X1)
    J2, K2 = size(X2)
    N2, L = size(D)
    @assert N==N2 && J==J2 
    return (N=N, J=J, K1=K1, K2=K2, L=L) 
end

function get_kdim(X1::Array{T,3}, X2::Matrix{T}, D::Matrix{T}) where T
    N, J, K1, K2 = get_dim(X1, X2, D)
    kdim = K1+K2+J-1
    return get_kdim
end
