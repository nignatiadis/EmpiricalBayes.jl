try
    using RCall
end
using RCall
using Distributions
using StatsBase
using Iterators
using Gurobi
using JLD

using EmpiricalBayes


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

i = parse(Int64, ARGS[1])
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

dry_run = CEB_ci(rand(d_true,Int(m/2)), rand(d_true, Int(m/2)), ds, target)

dry_run_snoop = CEB_ci(rand(d_true,m), rand(d_true, m), ds, target)

for k=1:k_max
    sim_array = []
    for l = 1:nreps
        @show l
        Xs = rand(d_true,m)
        brad_non_null =  fit(BradDeconvolveR, Xs; deltaAt = :nothing)
        brad_null =  fit(BradDeconvolveR, Xs)

        n_total = length(Xs)
        n_half = ceil(Int, n_total/2)
        idx_test = sample(1:n_total, n_half, replace=false)
        idx_train = setdiff(1:n_total, idx_test)
        Xs_train = Xs[idx_train]
        Xs_test = Xs[idx_test]

        ceb_res = CEB_ci(Xs_train, Xs_test, ds, target,
                    dry_run[2], dry_run[3], dry_run[4].C_bias)
        ceb_res_snoop = CEB_ci(Xs, Xs, ds, target,
                    dry_run_snoop[2], dry_run_snoop[3], dry_run_snoop[4].C_bias)

        push!(sim_array, (ceb_res, ceb_res_snoop, brad_null, brad_non_null))
    end
    res = (i, comb, sim_array)
    save("/scratch/users/ignat/sims/May22/mysim_$(i)_$(k).jld", "res", res)
end
