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


## With LFSR Numerator

## First a case in which result is really what we do not want
## which happens when there is a large delta at 0.

estimate(brad_jl, PosteriorTarget(LFSRNumerator(0.0));debias=true)
confint(brad_jl, PosteriorTarget(LFSRNumerator(0.0)), 0.95, ;debias=false)


lfsr_riesz = riesz_representer.(LFSRNumerator(2.0), tau)
lfsr_riesz_at_0 = copy(lfsr_riesz)
lfsr_riesz_at_0[19] /=2
lfsr_riesz'*g_prob
lfsr_riesz_at_0'*g_prob

#
brad_jl_no0 = fit(BradDeconvolveR, prostz; c0=1.0,
                     deltaAt = :nothing)

lfsr_riesz_at_0'*brad_jl_no0.g_prior
lfsr_riesz'*brad_jl_no0.g_prior

# Difference still not negligible, let us use a much larger grid.


tau2 = collect(-3.6:0.002:3.6)
brad_jl_tau2 = fit(BradDeconvolveR, prostz; c0=0.0,
                     deltaAt = :nothing,  prior_grid=tau2)


lfsr_riesz = riesz_representer.(LFSRNumerator(2.0), tau2)
lfsr_riesz_at_0 = copy(lfsr_riesz)
lfsr_riesz_at_0[181] /=2

lfsr_riesz_at_0'*brad_jl_tau2.g_prior
lfsr_riesz'*brad_jl_tau2.g_prior

confint(brad_jl_tau2, LFSRNumerator(2.0), 0.95, ;debias=false)
# Discretization error here is larger than confindence interval length. Ugh.


# Let us..

tau2
# OK need to rework this from scratch more or less..
# Idea: Make grid finer and properly integrate from a bit left to right.


# TODO: 1) Implement the below
# 2) Check if method gives correct coverage in Normal/Normal example
# 3) Check how method behaves in our other two examples which we want to simulate from..
function riesz_representer(target::LinearInferenceTarget, b::BradDeconvolveR)

end

# Afou ftasw kai isws koimh8w ligo ->
# prwta na dw ola ta experiment settings pou 8elw na tre3w
# Meta na dw pws leitourgei tou Brad, toulaxiston kapoies fores 8a 8elw swsto coverage
# Isws kai ena grhgoro peirama na dw ti coverage petyxainoume...

# Maybe even demonstrate everything on 1 simulation example....
