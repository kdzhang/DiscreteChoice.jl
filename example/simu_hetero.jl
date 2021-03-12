# In this simulation, we allow different products to have different mean utilities
# and also individual specific utility

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
β_mean = [1, -2, 3, -4, 5] .* 0.1
σ = [1, 2, 1, 2, 1] .* 0.1
β = zeros(N, K)
for i in 1:K
    β[:,i] = randn(N) .* σ[i] .+ β_mean[i]
end

index = β*X' .+ ones(N)*vcat(0, ξ)' # N-by-J
Y = zeros(N)
y_id = 1:J
for i in 1:N
    y_prob = exp.(index[i,:]) ./ sum(exp.(index[i,:]))
    Y[i] = sample(y_id, Weights(y_prob))
end

y_prob_emp = [mean(Y.==i) for i in 1:J ]
@show y_prob .- y_prob_emp |> maximum

@save "example/data/hetero.jld2" Y X β_mean ξ σ

