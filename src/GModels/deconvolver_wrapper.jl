struct BradDeconvolveR <: EmpiricalBayesEstimator
    prior_grid::Vector{Float64}
    g_prior::Vector{Float64}
    g_bias::Vector{Float64}
    g_cov::Matrix{Float64}
end

struct SmoothedBradDeconvolveR <: EmpiricalBayesEstimator
    unsmoothed::BradDeconvolveR
    prior_σ::Float64
    prior_grid::Vector{Float64}
end

function fit(::Type{BradDeconvolveR}, Xs; c0=0.01,
           deltaAt = :nothing, pDegree=10, prior_grid=collect(-3.6:0.02:3.6))
    R"library(deconvolveR)"
    if deltaAt == :nothing
        R"result <- deconv(tau = $(prior_grid), X = $(Xs), c0=$(c0),
                   deltaAt = NULL, family = 'Normal', pDegree = $(pDegree))"
    else
        R"result <- deconv(tau = $(prior_grid), X = $(Xs), c0=$(c0),
                       deltaAt = $(deltaAt), family = 'Normal', pDegree = $(pDegree))"
    end
    R"result_mat <- result$stats"
    R"result_cov <- result$cov.g"
    @rget result_mat
    @rget result_cov
    g_prior = result_mat[:,2]
    g_bias = result_mat[:,6]
    g_cov = result_cov
    BradDeconvolveR(prior_grid, g_prior, g_bias, g_cov)
end

function estimate(brad::BradDeconvolveR, target::LinearInferenceTarget; debias=true)
    gs = brad.g_prior
    g_bias = debias ? brad.g_bias : zero(gs)
    lin_coef = riesz_representer.(target,  brad.prior_grid)
    lin_coef'*(gs .+ g_bias)
end

function StatsBase.confint(brad::BradDeconvolveR, target::LinearInferenceTarget, alpha::Float64=0.1; kwargs...)
    point_est = estimate(brad, target; kwargs...)
    lin_coef = riesz_representer.(target,  brad.prior_grid)
    sd = sqrt(lin_coef'*brad.g_cov*lin_coef)
    q = quantile(Normal(), 1-alpha/2)
    (point_est - q*sd, point_est + q*sd)
end


# actually make the below generic for any type of EBayes thing
function estimate(brad::Union{BradDeconvolveR, SmoothedBradDeconvolveR}, target::PosteriorTarget; kwargs...)
    num = estimate(brad, target.num; kwargs...)
    denom = estimate(brad, target.denom; kwargs...)
    num/denom
end


function StatsBase.confint(brad::BradDeconvolveR, target::PosteriorTarget, alpha::Float64=0.1; debias=true)
    num_est = estimate(brad, target.num; debias=debias)
    denom_est = estimate(brad, target.denom; debias=debias)
    point_est = num_est/denom_est

    num_coef = riesz_representer.(target.num,  brad.prior_grid)
    denom_coef = riesz_representer.(target.denom,  brad.prior_grid)

    delta_method_coef = num_coef./denom_est .- denom_coef.*num_est./(denom_est^2)

    sd = sqrt(delta_method_coef'*brad.g_cov*delta_method_coef)

    q = quantile(Normal(), 1-alpha/2)

    (point_est - q*sd, point_est + q*sd)
end

function estimate(brad::Union{BradDeconvolveR, SmoothedBradDeconvolveR},
                  target::InferenceTarget, Xs; debias=true)
    estimate(brad, target; debias=debias)
end

#----------------------------------------------------------
# Fitting and estimation functions for SmoothedDeconvolveR
#----------------------------------------------------------

function fit(::Type{SmoothedBradDeconvolveR}, Xs, prior_σ = 0.02;
              prior_grid=collect(-3.6:0.02:3.6), kwargs...)
    marginal_σ = sqrt(1.0 + prior_σ^2)
    Xs_rescaled = Xs./marginal_σ
    deconv = fit(BradDeconvolveR, Xs_rescaled;
                 prior_grid = prior_grid./marginal_σ, kwargs...)
    SmoothedBradDeconvolveR(deconv, prior_σ, prior_grid)
end

function MixtureModel(smooth_g::SmoothedBradDeconvolveR)
   prior_σ = smooth_g.prior_σ
   normals = [Normal(μ, prior_σ) for μ in smooth_g.prior_grid]
   MixtureModel(normals, smooth_g.unsmoothed.g_prior)
end

function smoothed_riesz(smooth_g::SmoothedBradDeconvolveR, target::LinearInferenceTarget)
    mm = MixtureModel(smooth_g)
    posterior_stats.(mm.components, target)
end
function estimate(brad::SmoothedBradDeconvolveR, target::LinearInferenceTarget; debias=true)
    gs = brad.unsmoothed.g_prior
    g_bias = debias ? brad.unsmoothed.g_bias : zero(gs)
    lin_coef = smoothed_riesz(brad, target)
    lin_coef'*(gs .+ g_bias)
end

function confint(brad::SmoothedBradDeconvolveR, target::LinearInferenceTarget, alpha::Float64=0.1; kwargs...)
    point_est = estimate(brad, target; kwargs...)
    lin_coef = smoothed_riesz(brad,  target)
    sd = sqrt(lin_coef'*brad.unsmoothed.g_cov*lin_coef)
    q = quantile(Normal(), 1-alpha/2)
    (point_est - q*sd, point_est + q*sd)
end

function confint(brad::SmoothedBradDeconvolveR, target::PosteriorTarget, alpha::Float64=0.1; debias=true)
    num_est = estimate(brad, target.num; debias=debias)
    denom_est = estimate(brad, target.denom; debias=debias)
    point_est = num_est/denom_est

    num_coef = smoothed_riesz(brad, target.num)
    denom_coef = smoothed_riesz(brad,  target.denom)

    delta_method_coef = num_coef./denom_est .- denom_coef.*num_est./(denom_est^2)

    sd = sqrt(delta_method_coef'*brad.unsmoothed.g_cov*delta_method_coef)

    q = quantile(Normal(), 1-alpha/2)

    (point_est - q*sd, point_est + q*sd)
end
