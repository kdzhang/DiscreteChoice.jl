using DiscreteChoice
using JLD2
using ForwardDiff
using Optim

@load "example/data/homo.jld2" Y X β ξ
@load "example/data/hetero.jld2" Y X β_mean ξ σ

@code_warntype llk_sample(Y, X, β, ξ)


ccp(Y,X)

llk_sample(Y::Vector, X::Matrix, β_mean::Vector, ξ::Vector, σ::Vector; n_sim = 100)


res_homo = opt_homo(Y, X, vcat(β, ξ))
res_hetero = opt_hetero(Y, X, vcat(β_mean, ξ, σ); n_sim=5)


Optim.minimizer(res_homo) .- vcat(β, ξ)
Optim.minimizer(res_hetero) .- vcat(β_mean, ξ, σ)

f = para -> -llk_sample(Y, X, para[1:K], para[K+1:end]) # take negative to minimize
g = TwiceDifferentiable(f, init_guess; autodiff=:forward)


K = 5
J = 10
f_try = para -> -llk_sample(Y, X, para[1:K], para[K+1:K+J-1], para[K+J:end]; n_sim=100)
ForwardDiff.gradient(f_try, vcat(β_mean, ξ, σ))