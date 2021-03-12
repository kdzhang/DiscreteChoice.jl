using DiscreteChoice
using JLD2
using ForwardDiff
using Optim

@load "example/data/homo.jld2" Y X β ξ

model = DCModel(Y,X; hetero_preference=false)

opt_main!(model; init_guess = vcat(β, ξ))

res = Optim.minimizer(model.optResults) |> copy
res .- vcat(β, ξ) |> maximum