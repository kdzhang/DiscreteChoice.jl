using StatsBase
using ForwardDiff
using Optim

"Homogeneous Preference MLE"
function llk_sample(Y::Vector, X::Matrix, β::Vector, ξ::Vector)
    # Y: N-by-1
    # X: J-by-K
    # β: K-by-1
    # ξ: (J-1)-by-1, first normalized to 0

    ξ = vcat(zero(eltype(ξ)), ξ)
    index = exp.(X*β .+ ξ)
    y_prob = index ./ sum(index)
    llk = 0.0
    for i in eachindex(Y)
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

"heterogeneous Preference"
function llk_sample(Y::Vector, X::Matrix, β_mean::Vector{T}, ξ::Vector{T}, σ::Vector{T};
    n_sim = 100) where T<:Real
    # Y: N-by-1
    # X: J-by-K
    # β: K-by-1
    # ξ: (J-1)-by-1, first normalized to 0

    N = length(Y)
    J, K = size(X)

    Y = Int.(Y)
    ξ = vcat(zero(eltype(ξ)), ξ)
    # @show size(ξ)

    llk = 0.0
    for i in 1:N
        β_ind = zeros(T, n_sim, K)
        for k in 1:K
            β_ind[:,k] = randn(n_sim) .* σ[k] .+ β_mean[k]
            # β_ind[:,k] .=  β_mean[k]
        end
        index = exp.(β_ind*X' .+ ones(n_sim)*ξ') # n_sim-by-J

        y_prob = index ./ (sum(index, dims=2) * ones(1,J)) # nsim-by-J
        y_prob = mean(y_prob, dims=1)
        # @show size(y_prob)
        llk += log(y_prob[Y[i]])
    end

    return llk
end



function opt_homo(Y, X, init_guess)
    J, K = size(X)
    N = length(Y)
    f = para -> -llk_sample(Y, X, para[1:K], para[K+1:end]) # take negative to minimize
    # g = TwiceDifferentiable(f, init_guess; autodiff=:forward)
    # opt = Optim.optimize(g, init_guess)
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


function opt_main!(model::DCModel; init_guess::Vector, n_sim=5)
    if !model.hetero_preference
        model.optResults = opt_homo(model.Y, model.X, init_guess)
    else 
        model.optResults = opt_hetero(model.Y, model.X, init_guess, n_sim)
    return model
end
