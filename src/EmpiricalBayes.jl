module EmpiricalBayes

using Distributions
using StatsBase
using QuadGK
using HCubature
using Gurobi
using JuMP
using Roots
using TSVD
using SpecialFunctions
using KernelDensity
using RCall
using LaTeXStrings
using LinearAlgebra
using Random
using RecipesBase

import Base: length
import Base.Broadcast: broadcastable
import Distributions: pdf, estimate, cf, MixtureModel, ContinuousUnivariateDistribution
import StatsBase: fit, confint
import RCall: rcopy, RClass, rcopytype

include("types.jl")
include("utils.jl") # add tests
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
include("FModels/brown_greenshtein.jl")
include("plotrecipes.jl")


export NormalConvolutionProblem,
       LinearEstimator,
       EmpiricalBayesEstimator,
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
       maxbias,
       donoho_ci,
       CalibratedCI,
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
       DeLaValleePoussinKernel,
       sinc_kde,
       BinnedMarginalDensityNeighborhood,
       CEB_ci,
       CEB_ci_cb,
       BradDeconvolveR,
       SmoothedBradDeconvolveR,
       PriorTailProbability,
       OneSidedPriorTailProbability,
       BrownGreenshtein,
       KDECalibrator,
       inference_target,
       MarginalDistributionTarget,
       pretty_label
end # module
