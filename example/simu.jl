using DiscreteChoice
using Optim
using ForwardDiff
using Random, StatsBase
using ProfileView

#################
# Simulate data
#################

Random.seed!(42)

N = 20000
K1 = 1
K2 = 5
L = 4
J = 10

Y = rand(1:J,N)
X1 = randn(N,J,K1)
X2 = randn(J,K2)
D = randn(N,L)

α = randn(K1)
Π = randn(K2, L)
ξ_all = vcat(0, rand(J-1))
ξ = ξ_all[2:end]

Py = DiscreteChoice.y_prob(X1,X2,D, α, Π, ξ_all)
Y = zeros(Int64, N)
for i in eachindex(Y)
    Y[i] = sample(1:J, Weights(Py[i,:]))
end
[mean(Y.==i) for i in 1:J]


@time llk_sample_fullset(Y, X1, X2, D, α, Π, ξ)

# Try with random choice subset
C = DiscreteChoice.get_random_set(J, 5, Y)
@time llk_sample_subset(Y, X1, X2, D, C, α, Π, ξ)

#######################
# Estimate the model
#######################

model_homo = DCModel(Y,X1,X2,D; hetero_preference=false)
estDCModel!(model_homo; init_guess = initial_guess(model_homo))
updateCoef!(model_homo)

model_homo.α .- α
model_homo.Π .- Π
model_homo.ξ .- ξ


# Notice that if we start from the true value, it gives correct results
# but if we start from random value, the results are very off, even for merely 10 parameters

para_true = vcat(α, Π[:], ξ)
model_homo_2 = DCModel(Y,X1,X2,D; hetero_preference=false)
# estDCModel!(model_homo_2; init_guess = initial_guess(model_homo_2), subset = C)
estDCModel!(model_homo_2; init_guess = para_true, subset = C)
updateCoef!(model_homo_2)

model_homo_2.α .- α
model_homo_2.Π .- Π
model_homo_2.ξ .- ξ