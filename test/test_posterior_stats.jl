# test different implementations for checking if >=


using EmpiricalBayes
using Test
using Random: seed!
using Distributions
import QuadGK:quadgk

marginal_grid = collect(range(-6,stop=6,length=1001));
prior_grid = collect(range(-3,stop=3,length=121));
ds = MixingNormalConvolutionProblem(Normal, 0.2, prior_grid, marginal_grid);

true_dist = MixtureModel([ Normal(-2,.2), Normal(+2,.2)])
d_center = NormalConvolutionProblem(true_dist, marginal_grid);
f_center = BinnedMarginalDensity(d_center)

# Compare current implementation to integrating over correct domain
t = LFSRNumerator(1.0)
posterior_stats(d_center, t)

# nice!
quadgk(β ->  pdf(d_center.prior, β)*riesz_representer(t, β) , 0,20)[1]
