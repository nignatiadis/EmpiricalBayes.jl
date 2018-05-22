using RCall
using Distributions
using StatsBase
using Iterators
using Gurobi
using JLD
using Base.Test
using EmpiricalBayes

using KernelDensity
nreps =20
k_max = 12 #in total 25*16 replications

true_dists = [MixtureModel([ Normal(-0.3,.2), Normal(0,0.9)],[0.8, 0.2]), EmpiricalBayes.ash_flattop ]
xs = collect(-3:0.5:3)
minimax_σs = [0.2, 0.5]
ms = [20_000, 100_000]

combs = []

for (true_dist, x, σ, m) in product(true_dists, xs, minimax_σs, ms)
    push!(combs, Dict(:dist=>true_dist, :x=>x, :σ=>σ, :m=>m) )
end

#i = parse(Int64, ARGS[1])
i=1
comb = combs[i]

m = comb[:m]
σ = comb[:σ]
true_dist = comb[:dist]
x = comb[:x]

marginal_grid = collect(linspace(-7.0,7.0,1001));
prior_grid = collect(linspace(-3,3,121));
marginal_h = marginal_grid[2]-marginal_grid[1];

d_true = NormalConvolutionProblem(true_dist, marginal_grid)
ds = MixingNormalConvolutionProblem(Normal, σ, prior_grid, marginal_grid);

target = PosteriorTarget(LFSRNumerator(x))

Xs_train = rand(d_true,Int(m/2))

Xs_test = rand(d_true, Int(m/2))

dry_run = CEB_ci(Xs_train, Xs_test, ds, target)

m_train = length(Xs_train)
m_test = length(Xs_test)

marginal_grid = ds.marginal_grid
marginal_h = ds.marginal_h

M_bd = marginal_h/sqrt(2*pi)

f_const = BinnedMarginalDensity(M_bd, marginal_grid, marginal_h)
# Train
# Estimate the numerator
M_max_num = MinimaxCalibrator(ds, f_const, m_train, target.num;
             C=Inf);


# Estimate the denominator
M_max_denom = MinimaxCalibrator(ds, f_const, m_train, target.denom;
            C=Inf);

f_nb = fit(BinnedMarginalDensityNeighborhood, Xs_train, ds)

C_bias = f_nb.C_bias

f_nb.f_kde

CEB_ci(Xs_train, Xs_test, ds, target,
            M_max_num,
            M_max_denom,
            C_bias;
            C=:auto, conf=0.1)


tmp1 = sinc_kde(Xs_train, marginal_grid)


marginal_grid2 = collect(linspace(-6.0,6.0,1001));
tmp2= sinc_kde(Xs_train, marginal_grid2)

marginal_h
tmp3 = kde(Xs_train, -7:(marginal_h-1e-10):7)

marginal_h2 = marginal_grid2[2]-marginal_grid2[1]
tmp4 = kde(Xs_train, -6:marginal_h2:6)
