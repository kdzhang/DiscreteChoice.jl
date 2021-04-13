function y_prob(X1::Array{T,3}, X2::Matrix{T}, D::Matrix{T}, α::Vector{T}, Π::Matrix{T}, 
    ξ_all::Vector{T}) where T
    N, J, K1, K2, L = get_dim(X1, X2, D)
    @assert length(ξ_all) == J
    y_prob = zeros(T, N, J)
    for (i, d) in enumerate(eachrow(D))
        index = exp.(X2*(Π*d) .+ X1[i,:,:]*α .+ ξ_all) # J-by-1
        y_prob[i,:] = index ./ sum(index)
    end
    return y_prob
end

# TODO: need to add a y_prob for hetero preference y_prob



"""
Sample likelihood when consumers face homogeneous choice set, no unobservable preference heterogeneity
"""
function llk_sample_fullset(Y::Vector, X1::Array, X2::Matrix, D::Matrix, 
    α::Vector{T}, Π::Matrix{T}, ξ::Vector{T}) where T

    N, J, K1, K2, L = get_dim(X1, X2, D)
    ξ_all = vcat(zero(eltype(ξ)), ξ)
    @assert length(ξ_all) == J

    llk = 0.0
    for (i, d) in enumerate(eachrow(D))
        index = exp.(X2*(Π*d) .+ X1[i,:,:]*α .+ ξ_all) # J-by-1
        y_prob = index ./ sum(index)
        llk += log(y_prob[Y[i]])
    end
    return llk
end


"""
Sample likelihood when consumers face different choice sets, no unobservable preference heterogeneity

C::Matrix, N-by-J dummy matrix, determines the choice set of each individual
    This can be actual choice set restrictions, or purely random choice sets determined to facilitate estimation
"""
function llk_sample_subset(Y::Vector, X1::Array, X2::Matrix, D::Matrix, C::Vector{Vector{Int64}},
    α::Vector{T}, Π::Matrix{T}, ξ::Vector{T}) where T

    N, J, K1, K2, L = get_dim(X1, X2, D)
    ξ_all = vcat(zero(eltype(ξ)), ξ)
    @assert length(ξ_all) == J

    llk = 0.0
    for (i, d) in enumerate(eachrow(D))
        choice_id = findfirst(isequal(Y[i]), C[i])
        index = @views exp.(X2[C[i],:]*(Π*d) .+ X1[i,C[i],:]*α .+ ξ_all[C[i]]) # J-by-1
        y_prob = index ./ sum(index)
        llk += log(y_prob[choice_id])
    end
    return llk
end

function llk_sample(Y::Vector, X1::Array, X2::Matrix, D::Matrix, 
    α::Vector{T}, Π::Matrix{T}, ξ::Vector{T}; subset = nothing) where T

    if isnothing(subset)
        llk = llk_sample_fullset(Y, X1, X2, D, α, Π, ξ)
    else
        llk = llk_sample_subset(Y, X1, X2, D, subset,  α, Π, ξ)
    end
    return llk
end


"""
Sample likelihood when there are unobservable preference heterogeneity
"""
function llk_sample(Y::Vector, X1::Array, X2::Matrix, D::Matrix, 
    α::Vector{T}, Π::Matrix{T}, ξ::Vector{T}, σ::Vector{T}, n_sim::Integer) where T

    N, J, K1, K2, L = get_dim(X1, X2, D)
    ξ_all = vcat(zero(eltype(ξ)), ξ)
    @assert length(ξ_all) == J

    llk = 0.0
    for (i, d) in enumerate(eachrow(D))
        y_prob = zeros(T, n_sim, J)
        for i_sim = 1:n_sim
            index = exp.(X2*(Π*d .+ randn(K).*σ) .+ X1[i,:,:]*α .+ ξ_all) # J-by-1
            y_prob[i_sim,:] = index ./ sum(index)
        end
        y_prob = mean(y_prob, dims=1)
        llk += log(y_prob[Y[i]])
    end
    return llk
end


"""
CCP approach for homogeneous preference

    Note that the CCP method will only work for large number of commodities.
    Also, we will need IV for ξ to control endogeneity
"""
function ccp(Y::Vector, X::Matrix)
    # Y: N-by-1
    # X: J-by-K
    J, K = size(X)
    P = proportionmap(Y)
    δ = [P[j]-P[1] for j in 2:J]
    DX = zeros(J-1,K)
    for j = 2:J-1
        DX[j,:] = X[j,:] .- X[1,:]
    end
    β_hat = (DX'*DX)\(DX'*δ)
    return β_hat
end



function opt_homo(Y, X1, X2, D, init_guess; subset = nothing)
    N, J, K1, K2, L = get_dim(X1, X2, D)
    if isnothing(subset)
        f = para -> -llk_sample_fullset(Y, X1, X2, D, 
            para[1:K1], # α 
            reshape(para[K1+1:K1+K2*L], K2, L), # Π
            para[K1+K2*L+1:end]) # ξ
            # take negative to minimize
    else
        f = para -> -llk_sample_subset(Y, X1, X2, D, subset,
            para[1:K1], # α 
            reshape(para[K1+1:K1+K2*L], K2, L), # Π
            para[K1+K2*L+1:end]) # ξ
            # take negative to minimize
    end
    opt = Optim.optimize(f, init_guess, LBFGS(); autodiff=:forward)
    return opt
end


function opt_hetero(Y, X, init_guess, n_sim)
    J, K = size(X)
    N = length(Y)
    f = para -> -llk_sample(Y, X, para[1:K], para[K+1:K+J-1], para[K+J:end]; n_sim=n_sim) # take negative to minimize, ξ only J-1 dim
    # g = TwiceDifferentiable(f, init_guess; autodiff=:forward)
    opt = Optim.optimize(f, init_guess, LBFGS(); autodiff=:forward)
    return opt
end


function estDCModel!(model::DCModel; init_guess::Vector, n_sim=5, subset=nothing)
    if !model.hetero_preference
        model.optResults = opt_homo(model.Y, model.X1, model.X2, model.D, init_guess; subset=subset)
    else 
        model.optResults = opt_hetero(model.Y, model.X, init_guess, n_sim)
    end
    return model
end
