using Plots
using Test
using RCall
using EmpiricalBayes
using StatsBase
using Distributions

R"load('./notebooks/datasets/prostz.Rdata')"

@rget prostz
R"library(deconvolveR)"

R"tau <- seq(-3.6,3.6, by = 0.2)"
@rget tau
R"result <- deconv(tau = tau, X = prostz, deltaAt = 0, family = 'Normal', pDegree = 5)"
R"result_mat <- result$stats"
@rget result_mat

#result_mat[18:20, 2:3]
#plot(tau, result_mat[:,2])
#tau_nz = copy(tau)
#deleteat!(tau_nz, 19)
#gs_nz = result_mat[:,2]
#deleteat!(gs_nz, 19)
#plot(tau_nz, gs_nz)

R"result_cov <- result$cov.g"
@rget result_cov

g_prob = result_mat[:,2]
g_bias = result_mat[:,6]

prob_greater_1 = sum(g*(abs(t) >1) for (t,g) in zip(tau,g_prob) )
prob_greater_1 = sum(g*(abs(t) >1) for (t,g) in zip(tau,g_prob) )

greater_1_linearization = one(Float64).*(abs.(tau) .>1)
greater_1_linearization'*(g_prob .+ g_bias)
greater_1_linearization'*g_prob

prob_greater_1_ci = 1.96*sqrt(greater_1_linearization'*result_cov*greater_1_linearization)

greater_eq_2_linearization = one(Float64).*(abs.(tau) .>=2)
greater_eq_2_linearization'*g_prob
prob_greater_2_ci = 1.96*sqrt(greater_eq_2_linearization'*result_cov*greater_eq_2_linearization)

# Why is this factor 2 off efron's result?


R"result_P <- result$P"
@rget result_P

se_1 = diag(result_cov).^0.5
se_2 = result_mat[:, 3]

@test se_1 == se_2

R"class(result) <- 'deconvolveR'"
R"class(result)"

a = R"result"


brad_jl = fit(BradDeconvolveR, prostz; c0=1.0)

@test greater_eq_2_linearization'*g_prob ≈ estimate(brad_jl, PriorTailProbability(2.0);debias=false)

(l1,r1) = EmpiricalBayes.confint(brad_jl, PriorTailProbability(2.0), 0.95; debias=false)
r1-l1

estimate(brad_jl, PosteriorTarget(PosteriorMeanNumerator(2.0));debias=false)

confint(brad_jl, PosteriorTarget(PosteriorMeanNumerator(2.0)), 0.95, ;debias=false)
