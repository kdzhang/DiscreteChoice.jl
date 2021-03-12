mutable struct DCModel
    Y::Vector{Int32}
    X::Matrix{Float64} # does not include constant column

    # Parameters
    hetero_preference::Bool

    β::Vector{Float64} 
    ξ::Vector{Float64} 
    σ::Vector{Float64} 

    optResults
end

function DCModel(Y, X; hetero_preference)
    Y = Int.(Y)
    N = length(Y)
    J, K = size(X)

    # Initialize coef
    β = zeros(eltype(X), K)
    ξ = zeros(eltype(X), J-1)
    if hetero_preference
        σ = zeros(eltype(X), K)
    else 
        σ = zeros(eltype(X), 0)
    end

    optResults = nothing

    DCModel(Y,X, hetero_preference, β,ξ,σ,optResults)
end