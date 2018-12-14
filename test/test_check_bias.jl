using EmpiricalBayes
using Test
using Distributions

true_dist = MixtureModel([ Normal(-0.3,.5), Normal(1.05,.5)])
marginal_grid = collect(range(-6, stop=6, length=1001));

marginal_h = marginal_grid[2] - marginal_grid[1] # should really replace by range

prior_grid = collect(range(-3, stop=3, length=121));
d_true = NormalConvolutionProblem(true_dist, marginal_grid);

ds = MixingNormalConvolutionProblem(Normal, 0.2, prior_grid, marginal_grid);

m = 200

cb_cb = ComteButucea(MarginalDensityTarget(0.0), m, marginal_grid)
cb_kdecalib = KDECalibrator(SincKernel, m, marginal_grid)

max_bias_cb = check_bias(cb_cb, ds; maximization=true)

max_bias_cb_kdecalib = check_bias(cb_kdecalib, ds; maximization=true)

@test max_bias_cb ≈ max_bias_cb_kdecalib atol=0.0001
