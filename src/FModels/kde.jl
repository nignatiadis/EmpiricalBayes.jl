# Basic way of constructing Neighborhoods.

struct KDECalibrator
    kernel::ContinuousUnivariateDistribution
    Q::BinnedCalibrator
    m::Int64
end

function KDECalibrator(::Type{T}, m::Int64, marginal_grid;
    h= default_bandwidth(T, m)) where T<:ContinuousUnivariateDistribution
    kernel = T(h)
    Qc = BinnedCalibrator(marginal_grid, pdf.(kernel, marginal_grid))
    KDECalibrator(kernel, Qc, m)
end

# TODO: Make more general maybe
inference_target(a::KDECalibrator) = MarginalDensityTarget(0.0)

struct SincKernel <: ContinuousUnivariateDistribution
    h::Float64 #Bandwidth
end

default_bandwidth(a::Type{SincKernel}, m) = 1/sqrt(log(m))

cf(a::SincKernel, t) = one(Float64)*(-1/a.h <= t <= 1/a.h)

function pdf(a::SincKernel, t)
    if t==zero(Float64)
        return(one(Float64)/pi/a.h)
    else
        return(sin(t/a.h)/pi/t)
    end
end

struct DeLaValleePoussinKernel <: ContinuousUnivariateDistribution
    h::Float64 #Bandwidth
end

default_bandwidth(a::Type{DeLaValleePoussinKernel}, m) = 1.3/sqrt(log(m))

function cf(a::DeLaValleePoussinKernel, t)
    if abs(t * a.h) <= 1
        return(one(Float64))
    elseif abs(t * a.h) <= 2
        return(2*one(Float64) - abs(t * a.h))
    else
        return(zero(Float64))
    end
end

function pdf(a::DeLaValleePoussinKernel, t)
    if t==zero(Float64)
        return(3*one(Float64)/2/pi/a.h)
    else
        return(a.h*(cos(t/a.h)-cos(2*t/a.h))/pi/t^2)
    end
end

function sinc_kde(Xs, marginal_grid, ::Type{T};
     ws=KernelDensity.UniformWeights(length(Xs))) where T<:ContinuousUnivariateDistribution
    m = length(Xs)
    h = default_bandwidth(T,m)
    ker = T(h)
    (grid_min, grid_max) = extrema(marginal_grid)
    # hack for now to avoid weird floating point tricks wherein the conversion from
    # vector to range fails.. Should switch to ranges everywhere though
    marginal_h = marginal_grid[2] - marginal_grid[1] - 1e-10
    f_marginal = kde(Xs, ws,
         grid_min:marginal_h:grid_max, ker);
    f_marginal
end

function sinc_kde(Xs, marginal_grid; kwargs...)
    sinc_kde(Xs, marginal_grid, DeLaValleePoussinKernel; kwargs... )
end

mutable struct BinnedMarginalDensityNeighborhood
    f::BinnedMarginalDensity
    f_kde::UnivariateKDE
    C_bias::Float64
    C_std::Float64
end

# should be a fit function
function fit(::Type{BinnedMarginalDensityNeighborhood}, Xs,
                marginal_grid::Vector{Float64}, ::Type{T};
                nboot=101) where T<:ContinuousUnivariateDistribution

    marginal_h = marginal_grid[2] - marginal_grid[1]
    f_kde = sinc_kde(Xs, marginal_grid, T)

    m = length(Xs)
    C_stds = Vector{Float64}(undef, nboot)

    for k =1:nboot
        # Poisson bootstrap to estimate certainty band
        Z_pois = rand(Poisson(1), m)
        ws =  Weights(Z_pois/sum(Z_pois))
        f_kde_pois =  sinc_kde(Xs, marginal_grid, T; ws=ws)
        C_stds[k] = maximum(abs.(f_kde.density .- f_kde_pois.density))
    end

    C_std = median(C_stds)

    f = BinnedMarginalDensity(f_kde.density * marginal_h, marginal_grid, marginal_h)

    BinnedMarginalDensityNeighborhood(f, f_kde, 0.0, C_std)
end

function fit(::Type{BinnedMarginalDensityNeighborhood}, Xs,
       ds::MixingNormalConvolutionProblem, ::Type{T};
       kwargs...) where T<:ContinuousUnivariateDistribution

    m = length(Xs)
    marginal_grid = ds.marginal_grid

    f = fit(BinnedMarginalDensityNeighborhood, Xs, marginal_grid, T; kwargs...)

    #cb = ComteButucea(MarginalDensityTarget(0.0), m, marginal_grid)
    cb = KDECalibrator(T, m, marginal_grid)

    max_bias = check_bias(cb, ds; maximization=true)
    f.C_bias = max_bias
    f
end

function fit(::Type{BinnedMarginalDensityNeighborhood}, Xs,
       ds::MixingNormalConvolutionProblem;
       kwargs...)

       fit(BinnedMarginalDensityNeighborhood, Xs,
              ds::MixingNormalConvolutionProblem, DeLaValleePoussinKernel;
              kwargs...)
end

function check_bias(cb::Union{ComteButucea, KDECalibrator},
                  ds::MixingNormalConvolutionProblem;
                  maximization=true)

        f = BinnedMarginalDensity([],[],0)

        check_bias(cb.Q, ds, f, cb.m, inference_target(cb);
                         C=Inf, maximization=maximization)
end
