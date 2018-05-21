struct BradDeconvolveR
    prior_grid::Vector{Float64}
    g_prior::Vector{Float64}
    g_bias::Vector{Float64}
    g_cov::Matrix{Float64}
end

function fit(::Type{BradDeconvolveR}, Xs; c0=1.0,
           deltaAt = 0.0, pDegree=5, prior_grid=collect(-3.6:0.2:3.6))
    R"library(deconvolveR)"

    R"result <- deconv(tau = $(prior_grid), X = $(Xs), c0=$(c0),
                   deltaAt = $(deltaAt), family = 'Normal', pDegree = $(pDegree))"
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
    g_bias = debias?brad.g_bias:zeros(gs)
    lin_coef = riesz_representer.(target,  brad.prior_grid)
    lin_coef'*(gs .+ g_bias)
end

function StatsBase.confint(brad::BradDeconvolveR, target::LinearInferenceTarget, alpha::Float64=0.1; debias=true)
    point_est = estimate(brad, target; debias=debias)
    lin_coef = riesz_representer.(target,  brad.prior_grid)
    sd = sqrt(lin_coef'*brad.g_cov*lin_coef)
    q = quantile(Normal(), 1-alpha/2)
    (point_est - q*sd, point_est + q*sd)
end


function estimate(brad::BradDeconvolveR, target::PosteriorTarget; debias=true)
    num = estimate(brad, target.num; debias=debias)
    denom = estimate(brad, target.denom; debias=debias)
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
