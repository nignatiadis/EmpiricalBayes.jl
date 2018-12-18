using EmpiricalBayes
using Test
using Random: seed!

marginal_grid = collect(range(-6,stop=6,length=1001));
prior_grid = collect(range(-3,stop=3,length=121));
ds = MixingNormalConvolutionProblem(Normal, 0.2, prior_grid, marginal_grid);

true_dist = MixtureModel([ Normal(-2,.2), Normal(+2,.2)])
d_center = NormalConvolutionProblem(true_dist, marginal_grid);
f_center = BinnedMarginalDensity(d_center)

m = 1_000
ma = MinimaxCalibrator(ds, f_center, m, MarginalDensityTarget(0.0); C=Inf,
    ε=0.0, max_smoother=false);

seed!(1)
Xs = rand(true_dist, m)
