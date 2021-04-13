
mutable struct DCModel
    Y::Vector{Int32} # N-by-1, choices of individuals
    X1::Array{Float64,3} # N-J-K1: indiv-prod-char
    X2::Matrix{Float64} # J-K2: product char
    D::Matrix{Float64} # N-L: demographics
    C::Vector{Vector{Int64}} # N: choice set of each individual, 
                             # can be actual constraints or random draws to facilitate est

    # Parameters
    hetero_preference::Bool # unobservable hetero preference

    α::Vector{Float64} # K1: coef for X1
    Π::Matrix{Float64} # K2-L: coef for X2 interact with demo 
    ξ::Vector{Float64} # J-by-1: product FE, normalize the first commodity to 0
    σ::Vector{Float64} # K2-by-1: s.d. for unobs. hetero. preference for Π

    optResults # optimization results
end


function DCModel(Y, X1::Array{T,3}, X2::Matrix{T}, D::Matrix{T}; hetero_preference) where T
    Y = Int.(Y)
    N = length(Y)
    N2, J, K1 = size(X1)
    J2, K2 = size(X2)
    N3, L = size(D)
    
    @assert J==J2
    @assert N==N2==N3

    C = Vector{Vector{eltype(Y)}}(undef, N)

    # Initialize coef
    α = zeros(T, K1)
    Π = zeros(T, K2, L)
    ξ = zeros(T, J-1)
    if hetero_preference
        σ = zeros(T, K2)
    else 
        σ = zeros(T, 0)
    end

    optResults = nothing

    DCModel(Y,X1,X2,D,C, hetero_preference, α, Π, ξ, σ, optResults)
end

function updateCoef!(model::DCModel)
    @assert !isnothing(model.optResults)
    N,J,K1,K2,L = get_dim(model.X1, model.X2, model.D)

    res = Optim.minimizer(model.optResults)
    model.α = res[1:K1]
    model.Π = reshape(res[K1+1:K1+K2*L], K2, L)
    model.ξ = res[K1+K2*L+1:K1+K2*L+J-1] # ξ is only J-1 dim
    if model.hetero_preference
        model.σ = res[K1+K2*L+J+1:end]
    end
    return nothing
end
