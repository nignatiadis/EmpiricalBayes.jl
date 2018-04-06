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
include("GModels/npmle.jl")
include("donoho_minimax_calibrator.jl")
include("donoho_minimax_ci.jl")

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
       MinimaxCalibrator,
       NPMLE,
       check_bias,
       donoho_ci,
       DonohoCI
end # module
