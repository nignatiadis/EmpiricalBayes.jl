# Basic way of constructing Neighborhoods.


struct SincKernel <: Distributions.ContinuousUnivariateDistribution
    h::Float64 #Bandwidth
end

cf(a::SincKernel, t) = one(Float64)*(-1/a.h <= t <= 1/a.h)


function sinc_kde(Xs, marginal_grid;
     ws=KernelDensity.UniformWeights(length(Xs)))
    m = length(Xs)
    h = 1/sqrt(log(m))
    ker = SincKernel(h)
    (grid_min, grid_max) = extrema(marginal_grid)
    marginal_h = marginal_grid[2] - marginal_grid[1]
    f_marginal = kde(Xs, ws,
         grid_min:marginal_h:grid_max, ker);
    f_marginal
end


mutable struct BinnedMarginalDensityNeighborhood
    f::BinnedMarginalDensity
    f_kde::UnivariateKDE
    C_bias::Float64
    C_std::Float64
end

# should be a fit function
function fit(::Type{BinnedMarginalDensityNeighborhood}, Xs,
                marginal_grid::Vector{Float64}; nboot=101)

    marginal_h = marginal_grid[2] - marginal_grid[1]
    f_kde = sinc_kde(Xs, marginal_grid)

    m = length(Xs)
    C_stds = Vector{Float64}(nboot)

    for k =1:nboot
        # Poisson bootstrap to estimate certainty band
        Z_pois = rand(Poisson(1), m)
        ws =  Weights(Z_pois/sum(Z_pois))
        f_kde_pois =  sinc_kde(Xs, marginal_grid; ws=ws)
        C_stds[k] = maximum(abs.(f_kde.density .- f_kde_pois.density))
    end

    C_std = median(C_stds)

    f = BinnedMarginalDensity(f_kde.density * marginal_h, marginal_grid, marginal_h)

    BinnedMarginalDensityNeighborhood(f, f_kde, 0.0, C_std)
end

function fit(::Type{BinnedMarginalDensityNeighborhood}, Xs,
       ds::MixingNormalConvolutionProblem; kwargs...)

    m = length(Xs)
    marginal_grid = ds.marginal_grid

    f = fit(BinnedMarginalDensityNeighborhood, Xs, marginal_grid; kwargs...)

    cb = ComteButucea(MarginalDensityTarget(0.0), m, marginal_grid)
    max_bias = check_bias(cb,ds; maximization=true)
    f.C_bias = max_bias
    f
end

function check_bias(cb::ComteButucea,
                  ds::MixingNormalConvolutionProblem;
                  maximization=true)

        f = BinnedMarginalDensity([],[],0)

        check_bias(cb.Q, ds, f, cb.m, cb.target;
                         C=Inf, maximization=maximization)
end
