# Basic way of constructing Neighborhoods.


struct SincKernel <: Distributions.ContinuousUnivariateDistribution
    h::Float64 #Bandwidth
end

cf(a::SincKernel, t) = one(Float64)*(-1/a.h <= t <= 1/a.h)


function sinc_kde(Xs, marginal_grid)
    m = length(Xs)
    h = 1/sqrt(log(m))
    ker = SincKernel(h)
    (grid_min, grid_max) = extrema(marginal_grid)
    marginal_h = marginal_grid[2] - marginal_grid[1]
    f_marginal = kde(Xs, KernelDensity.UniformWeights(m),
         grid_min:marginal_h:grid_max, ker);
    f_marginal
end
#f_marginal_3 = kde(Xs_train, KernelDensity.UniformWeights(length(Xs_train)), -6:marginal_h:6, tmp_dist);

#Z_pois = rand(Poisson(1), length(Xs_train))
#f_marginal_4 = kde(Xs_train, Weights(Z_pois/sum(Z_pois)), -6:marginal_h:6, tmp_dist);
#f_kde4 = BinnedMarginalDensity(f_marginal_4.density * marginal_h, marginal_grid, marginal_h)
