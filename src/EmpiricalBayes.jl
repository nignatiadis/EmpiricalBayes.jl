module EmpiricalBayes

using Distributions
using StatsBase
using QuadGK
using Cubature
using Gurobi
using Convex
using Roots
using TSVD

import Base: length
import Distributions: pdf
import StatsBase: fit

include("bias_adjusted_ci.jl")
include("BayesProblem.jl")
include("PriorSets.jl")
include("f_modeling.jl")
include("donoho_minimax.jl")
include("GModels/npmle.jl")

export NormalConvolutionProblem,
       DiscretizedNormalConvolutionProblem,
       MixingNormalConvolutionProblem,
       marginal_grid_l,
       marginal_grid_r,
       convolution_matrix,
       posterior_stats,
       get_plus_minus,
       BinnedMarginalDensity,
       BinnedCalibrator,
       NPMLE
end # module
