using Distributions
using Iterators
using Gurobi
using JLD

using EmpiricalBayes


nreps = 50
true_dists = [MixtureModel([ Normal(-0.3,.5), Normal(1.05,.5)])]
ys = collect(linspace(0,3,7))
minimax_σs = [0.1, 0.5, 1.0]
ns = [20_000; 100_000]

combs = []

for (true_dist, y, σ, n) in product(true_dists, ys, minimax_σs, ns)
    push!(combs, Dict(:dist=>true_dist, :y=>y, :σ=>σ, :n=>n) )
end


i = parse(Int64, ARGS[1])

comb = combs[i]

n = comb[:n]
σ = comb[:σ]
true_dist = comb[:dist]
y = comb[:y]

marginal_grid = collect(linspace(-10,10,1001));
prior_grid = collect(linspace(-3,3,51));
d_true = NormalConvolutionProblem(true_dist, marginal_grid)
f = BinnedMarginalDensity(d_true)
m = n
ds = MixingNormalConvolutionProblem(Normal, 0.5, prior_grid, marginal_grid)

sim_array = []

for l = 1:nreps
    @show l
     Xs = rand(d_true,m)
     tst = donoho_ci(Xs, marginal_grid, prior_grid, y, σ, d_true)
     tst2 = donoho_ci(Xs, marginal_grid, prior_grid, y, σ, NPMLE)
     push!(sim_array, (tst, tst2))
end

res = (i, comb, sim_array)

save("/scratch/users/ignat/sims/Apr5/mysim_$i.jld", "res", res)
