using RCall
using Distributions
using StatsBase
using IterTools
using Gurobi
using JLD2

using EmpiricalBayes


nreps = 1
k_max = 4 #in total 25*16 replications
ks_par = 1:100#1:10

true_dists = [EmpiricalBayes.easy_dist, EmpiricalBayes.hard_dist]
xs = collect(-3.0:0.5:3.0)
minimax_σs = [0.2]
ms = [10_000]
targets =[ [PosteriorTarget(PosteriorMeanNumerator(x)) for x in xs],
           [PosteriorTarget(LFSRNumerator(x)) for x in xs]]
combs = []

for (true_dist, target, σ, m) in IterTools.product(true_dists, targets, minimax_σs, ms)
    push!(combs, Dict(:dist=>true_dist, :target=>target, :σ=>σ, :m=>m ))
end

comb_reps = []
for (l, k_par) in IterTools.product(1:length(combs), ks_par)
    push!(comb_reps, (l,k_par))
end

j = parse(Int64, ARGS[1])
comb_rep = comb_reps[j]
i = comb_rep[1]
k_rep = comb_rep[2]
comb = combs[i]

m = comb[:m]
σ = comb[:σ]
true_dist = comb[:dist]
target = comb[:target]

marginal_grid = collect(range(-6,stop=6,length=1001));
prior_grid = collect(range(-3,stop=3,length=121));
marginal_h = marginal_grid[2]-marginal_grid[1];

d_true = NormalConvolutionProblem(true_dist, marginal_grid)
ds = MixingNormalConvolutionProblem(Normal, σ, prior_grid, marginal_grid);


for k=1:k_max
    sim_array = []
    for l = 1:nreps
        @show l
        # get the sample
        Xs = rand(d_true,m)
        m_total = length(Xs)
        m_test= ceil(Int, m_total/2)
        m_train = m_total - m_test
        idx_test = sample(1:m_total, m_test, replace=false)
        idx_train = setdiff(1:m_total, idx_test)
        Xs_train = Xs[idx_train]
        Xs_test = Xs[idx_test]


        brad_fit =  fit(SmoothedBradDeconvolveR, Xs, 0.05; c0=0.01,prior_grid = collect(-3.6:0.05:3.6))
        f_nb = fit(BinnedMarginalDensityNeighborhood, Xs_train, ds; nboot=400)

        calibrator_res = []
        for t in target
            @show t
            CB_num = ComteButucea(t.num, m_train, marginal_grid)
            CB_denom = ComteButucea(t.denom, m_train, marginal_grid)
            ceb_res =  CEB_ci(Xs_train, Xs_test,
                        ds,
                        t,
                        CB_num, #
                        CB_denom; f_nb=f_nb)
            push!(calibrator_res, ceb_res)
        end
        push!(sim_array, (calibrator_res, brad_fit))
    end
    res = (i, comb, sim_array)
    @save "/scratch/users/ignat/sims/base_sim_Jan10_v/mysim_$(i)_$(k)_$(k_rep).jld" res
end
