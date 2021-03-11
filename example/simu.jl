# In this simulation, we allow different products to have different mean utilities

using Random
using Distributions
using StatsBase
using JLD2

Random.seed!(42)

N = 10_000  # num of consumers
J = 10      # num of products
K = 5       # num of char dim

X = randn(J, K) # Product characteristics
ξ = rand(J-1)
β = [1, -2, 3, -4, 5] .* 0.1
@assert length(β) == K

index = X*β .+ vcat(0, ξ)
y_prob = exp.(index) ./ sum(exp.(index))
y_id = 1:J

Y = sample(y_id, Weights(y_prob), N)

y_prob_emp = [mean(Y.==i) for i in 1:J ]
y_prob .- y_prob_emp |> maximum

@save "example/data/homo.jld2" Y X β ξ

