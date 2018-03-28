module EmpiricalBayes

using Distributions
using StatsBase
using QuadGK
using Cubature

include("BayesProblem.jl")
include("PriorSets.jl")

export NormalConvolutionProblem,
       DiscretizedNormalConvolutionProblem,
       marginal_grid_l,
       marginal_grid_r,
       convolution_matrix,
       posterior_stats


end # module
