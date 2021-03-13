module DiscreteChoice

export greet, llk_sample, ccp, opt_homo, opt_hetero, opt_main!, 
    DCModel, updateCoef!

greet() = print("Hello World!")
# Write your package code here.

include("struct.jl")
include("util.jl")
include("main.jl")


end
