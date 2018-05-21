module EmpiricalBayes

using Distributions
using StatsBase
using QuadGK
using Cubature
using Gurobi
using Convex
using JuMP
using Roots
using TSVD
using SpecialFunctions
using KernelDensity
using RCall

import Base: length
import Distributions: pdf, estimate, cf
import StatsBase: fit, confint
import RCall: rcopy, RClass, rcopytype

include("bias_adjusted_ci.jl")
include("inference_targets.jl")
include("BayesProblem.jl")
include("PriorSets.jl")
include("f_modeling.jl")
include("GModels/deconvolver_wrapper.jl")

include("GModels/npmle.jl")
include("donoho_minimax_calibrator.jl")
include("FModels/comte_butucea.jl")
include("FModels/kde.jl")
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
       DonohoCI,
       ComteButucea,
       MarginalDensityTarget,
       PosteriorTarget,
       LFSRNumerator,
       riesz_representer,
       LinearInferenceTarget,
       InferenceTarget,
       PosteriorTarget,
       PosteriorMeanNumerator,
       GeneralPosteriorLinearTarget,
       donoho_test,
       CalibratedNumerator,
       donoho_test2,
       SincKernel,
       sinc_kde,
       BinnedMarginalDensityNeighborhood,
       CEB_ci,
       CEB_ci_cb,
       BradDeconvolveR,
       PriorTailProbability

end # module
