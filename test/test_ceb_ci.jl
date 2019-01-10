using EmpiricalBayes
using Test
using Distributions
using StatsBase

true_dist = MixtureModel([ Normal(-0.3,.5), Normal(1.05,.5)])
marginal_grid = collect(range(-6, stop=6, length=1001));

marginal_h = marginal_grid[2] - marginal_grid[1] # should really replace by range

prior_grid = collect(range(-3, stop=3, length=121));
d_true = NormalConvolutionProblem(true_dist, marginal_grid);

ds = MixingNormalConvolutionProblem(Normal, 0.2, prior_grid, marginal_grid);

m_train= 10_000


Xs_train = rand(d_true,m_train)
Xs_test = rand(d_true,m_train)

target = PosteriorTarget(PosteriorMeanNumerator(1.0))

M_bd = marginal_h/sqrt(2*pi)
f_const = BinnedMarginalDensity(M_bd, marginal_grid, marginal_h)
# Train
# Estimate the numerator
M_max_num = MinimaxCalibrator(ds, f_const, m_train, target.num; C=Inf)
M_max_denom = MinimaxCalibrator(ds, f_const, m_train, target.denom; C=Inf)

ceb_ci_1 =  CEB_ci(Xs_train, Xs_test,
            ds,
            target,
            M_max_num, #
            M_max_denom)


# Check if CB intervals work

CB_num = ComteButucea(target.num, m_train, marginal_grid)
CB_denom = ComteButucea(target.denom, m_train, marginal_grid)

ceb_ci_1_cb =  CEB_ci(Xs_train, Xs_test,
            ds,
            target,
            CB_num, #
            CB_denom)

# Both approaches perform remarkably similar in this case 
posterior_stats(d_true, target)[3] #true target
confint(ceb_ci_1, target)
confint(ceb_ci_1_cb, target)

ceb_ci_1_cb.est_target, ceb_ci_1.est_target

brad_g =  fit(BradDeconvolveR, Xs_train, prior_grid=range(-3.6, stop=3.6, length=361);
    deltaAt = :nothing, pDegree=10)

ceb_ci_brad = CEB_ci(Xs_train, Xs_test,ds, target, brad_g)

posterior_stats(d_true, target)[3]

confint(ceb_ci_brad, target)
ceb_ci_brad.est_target
ceb_ci_brad.calibrated_target
