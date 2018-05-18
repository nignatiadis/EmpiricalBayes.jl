# compare output from KernelDensity.jl with SINC kernel
# to what Comte-Butucea yields.
using EmpiricalBayes
using Base.Test
using Distributions


true_dist = MixtureModel([ Normal(-0.3,.5), Normal(1.05,.5)])
marginal_grid = collect(linspace(-6,6,1001));

marginal_h = marginal_grid[2] - marginal_grid[1] # should really replace by range

prior_grid = collect(linspace(-3,3,121));
d_true = NormalConvolutionProblem(true_dist, marginal_grid);

ds = MixingNormalConvolutionProblem(Normal, 0.2, prior_grid, marginal_grid);

m = 3000

Xs = rand(d_true, m)

f_sinc = sinc_kde(Xs, marginal_grid)

x_tst_1 = f_sinc.x[700]
f_tst_1 = f_sinc.density[700]

# Let us compare to the result from comte_butucea
f_cbt_1 = estimate(Xs, ComteButucea,
 MarginalDensityTarget(x_tst_1), marginal_grid)

@test f_cbt_1 ≈ f_tst_1 atol=0.001  
#f_marginal = kde(Xs, KernelDensity.UniformWeights(length(Xs_train)), -6:marginal_h:6, SincK);
