# compare output from KernelDensity.jl with SINC kernel
# to what Comte-Butucea yields. (should be the same)
# also look at how DeLaValleePoussin performs

using EmpiricalBayes
using Test
using Distributions


true_dist = MixtureModel([ Normal(-0.3,.5), Normal(1.05,.5)])
marginal_grid = collect(range(-6, stop=6, length=1001));

marginal_h = marginal_grid[2] - marginal_grid[1] # should really replace by range

prior_grid = collect(range(-3, stop=3, length=121));
d_true = NormalConvolutionProblem(true_dist, marginal_grid);

ds = MixingNormalConvolutionProblem(Normal, 0.2, prior_grid, marginal_grid);

m = 3000

Random.seed!(2)
Xs = rand(d_true, m)

#----------------------------------------
#---- Test with actual Sinc Kernel-------
#----------------------------------------
f_sinc = sinc_kde(Xs, marginal_grid, SincKernel)


x_tst_1 = f_sinc.x[700]
f_tst_1 = f_sinc.density[700]

# Let us compare to the result from comte_butucea
f_cbt_1 = estimate(Xs, ComteButucea,
  MarginalDensityTarget(x_tst_1), marginal_grid)

@test f_cbt_1 ≈ f_tst_1 atol=0.001

#----- Check default dispatch ---------------------------------
dv_kde = sinc_kde(Xs, marginal_grid, DeLaValleePoussinKernel)
dv_kde_auto_dispatch = sinc_kde(Xs, marginal_grid)
@test dv_kde.density == dv_kde_auto_dispatch.density

#--------------------------------------------------------------
#---- Tests with both Sinc and DeLaValleePoussin Kernels-------
#--------------------------------------------------------------
f_true = pdf.(d_true, marginal_grid)
true_C = maximum(abs.(f_true .- f_sinc.density))

f_nb = fit(BinnedMarginalDensityNeighborhood, Xs, marginal_grid)
f_nb.C_std

f_nb_ds = fit(BinnedMarginalDensityNeighborhood, Xs, ds)

Xs_test = rand(d_true, m)
full_CI = CEB_ci(Xs, Xs_test, ds, PosteriorTarget(PosteriorMeanNumerator(2.0)))

full_CI[1].ci_left, full_CI[1].ci_right


# Use output from first run to speed up 2nd run by factor 4!
full_CI_fast_comp = CEB_ci(Xs, Xs_test, ds, PosteriorTarget(PosteriorMeanNumerator(2.0)),
                            full_CI[2], full_CI[3], full_CI[4].C_bias)

full_CI_fast_comp[1].ci_left, full_CI_fast_comp[1].ci_right

@test full_CI[1].ci_left ≈ full_CI_fast_comp[1].ci_left atol=0.005
@test full_CI[1].ci_right ≈ full_CI_fast_comp[1].ci_right atol=0.005

full_CI_inf = CEB_ci(Xs, Xs_test, ds, PosteriorTarget(PosteriorMeanNumerator(2.0)); C=Inf)

full_CI_inf[1].ci_left, full_CI_inf[1].ci_right

posterior_stats(d_true,  PosteriorTarget(PosteriorMeanNumerator(2.0)))
