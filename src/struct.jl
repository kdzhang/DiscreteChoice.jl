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

function updateCoef!(model::DCModel)
    @assert !isnothing(model.optResults)
    N = length(model.Y)
    J, K = size(model.X)

    res = Optim.minimizer(model.optResults)
    model.β = res[1:K]
    model.ξ = res[K+1:K+J-1] # ξ is only J-1 dim
    if model.hetero_preference
        model.σ = res[K+J:end]
    end
    return nothing
end
