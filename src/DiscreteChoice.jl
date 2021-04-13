module DiscreteChoice

using StatsBase
using ForwardDiff, Optim

export llk_sample, ccp, opt_homo, opt_hetero, 
    estDCModel!, initial_guess,
    DCModel, updateCoef!

# Write your package code here.

include("struct.jl")
include("util.jl")
include("main.jl")


end
