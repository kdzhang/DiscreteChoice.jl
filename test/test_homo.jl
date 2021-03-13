using DiscreteChoice
using JLD2
using ForwardDiff
using Optim
using StatsBase

@load "example/data/homo.jld2" Y X β ξ

P = zeros(10)
[P[i] = mean(Y.==i) for i in 1:10]
sum(P)

function y_prob(X, β, ξ)
    ξ = vcat(zero(eltype(ξ)), ξ)
    index = exp.(X*β .+ ξ)
    y_prob = index ./ sum(index)
    return y_prob
end



model_1 = DCModel(Y, X; hetero_preference=false)
model_2 = DCModel(Y, X; hetero_preference=false)

# Start from true value
opt_main!(model_1; init_guess = vcat(β, ξ))
abs.(Optim.minimizer(model_1.optResults) .- vcat(β, ξ)) |> maximum
updateCoef!(model_1)

P .- y_prob(model_1.X, model_1.β, model_1.ξ)

# Start from random value
opt_main!(model_2; init_guess = randn(length(β)+length(ξ)))
updateCoef!(model_2)

y_emp = y_prob(model_2.X, model_2.β, model_2.ξ)
P .- y_emp 
# Notice that this can perfectly match the data. Why? 
# simple reason is that the number of products is too small !!
# Because there are only 10 products, essentially only 10 data points, too many dimensions
# This will get changed if we include individual demographics into the data

abs.(Optim.minimizer(model_2.optResults) .- vcat(β, ξ)) |> maximum
Optim.minimizer(model_2.optResults) .- vcat(β, ξ)

β
ξ
# Why does the random starting value performs poorly? Is this some non-identification issue?

res_homo = opt_homo(Y, X, vcat(β, ξ))



