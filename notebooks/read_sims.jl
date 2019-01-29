using DataFrames

using Plots
using JLD2

using Distributions
using StatsBase
using IterTools
using EmpiricalBayes
using KernelDensity
using LaTeXStrings
using RCall
using FileIO


brad_degs = collect(1:15)



n_settings = 8

j = parse(Int64, ARGS[1])

par_comb = []
for (set,k) in IterTools.product(1:n_settings, 1:6)
    push!(par_comb, Dict(:setting => set, :k=>k))
end

s = par_comb[j][:setting]
setting_idx = s
k = par_comb[j][:k]

#set_idx = [Regex(string("mysim_",x, )) for x in 1:n_settings];

set_idx = Regex(string("mysim_",s,"_",k,"_"))

marginal_grid = collect(range(-6,stop=6,length=1001));
prior_grid = collect(range(-3,stop=3,length=121));
marginal_h = marginal_grid[2]-marginal_grid[1];

true_dists = [EmpiricalBayes.easy_dist, EmpiricalBayes.hard_dist]

#true_dists = zip( true_dists, d_trues)
xs = collect(-3.0:0.25:3.0)
minimax_σs = [0.2]
ms = [10_000; 40_000]
targets =[ [PosteriorTarget(PosteriorMeanNumerator(x)) for x in xs],
           [PosteriorTarget(LFSRNumerator(x)) for x in xs]]
combs = []

for (true_dist, target, σ, m) in IterTools.product(true_dists, targets, minimax_σs, ms)
    push!(combs, Dict(:dist=>true_dist, :target=>target, :σ=>σ, :m=>m ))
end

my_df = DataFrame( setting_idx = Int64[],
                   true_theta = Float64[],
                   method = String[],
                   dof = Int64[],
                   x=Float64[],
                   est = Float64[],
                   ci_left = Float64[],
                   ci_right = Float64[],
                   max_bias = Float64[])

#for setting_idx=1:n_settings
    res_list = readdir("/scratch/users/ignat/sims/base_sim_Jan10_v2");
    res_list = res_list[occursin.(Ref(set_idx), res_list)]

    # just to get d_true once
    file = string("/scratch/users/ignat/sims/base_sim_Jan10_v2/", res_list[1]);
    res = FileIO.load(file,"res");
    setting_idx = res[1]
    combo = res[2]
    d_true = NormalConvolutionProblem(combo[:dist], marginal_grid);

    for f in 1:length(res_list)
        @show f
        file = string("/scratch/users/ignat/sims/base_sim_Jan10_v2/", res_list[f]);
        res = FileIO.load(file,"res");
        setting_idx = res[1]
        combo = res[2]
        res_details = res[3][1]
        brad_combs = res_details[2]
        calib_vec = res_details[1]
        for l in 1:length(calib_vec)
            calib_x = calib_vec[l]
            t = combo[:target][l]
            x = t.num.x
            true_θ = posterior_stats(d_true,t)[3]
            #efron_est = estimate(brad,t)
            calib_est = estimate(calib_x, t)
            calib_ci_left, calib_ci_right = confint(calib_x,t)
            max_bias = maxbias(calib_x, t)
            #efron_ci_left, efron_ci_right = confint(brad, t)
            push!(my_df, (setting_idx, true_θ, "Calibrator", 0, x,
                         calib_est, calib_ci_left, calib_ci_right, max_bias))

            for d in 1:length(brad_degs)
                #@show d
                brad = brad_combs[d]
                d_b = brad_degs[d]
                efron_est = estimate(brad,t; debias=false)
                efron_ci_left, efron_ci_right = confint(brad, t; debias=false)
                efron_max_bias = maxbias(brad, t)
                push!(my_df, (setting_idx, true_θ, "Efron", d_b, x,
                       efron_est, efron_ci_left, efron_ci_right, efron_max_bias))
            end
        end
    end
#end

@save "/scratch/users/ignat/sims/base_sim_Jan10_v2/dfs/df_$(s)_$(k).jld2" my_df

#typeof(calib_x)
#DataFrame(setting_idx = res[1])
